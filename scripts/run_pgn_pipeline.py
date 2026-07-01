#!/usr/bin/env python3
import os
import time
import json
import glob
from pathlib import Path
import pandas as pd
import tensorflow as tf

from core.transfer_attack_core import (
    ATTACKER_MODELS,
    build_attacker,
    configure_cpu_runtime,
    denormalize,
    load_and_preprocess,
    resolve_image_path,
    run_attack,
    compute_embedding,
    save_adv,
)
from core.ir152_loader import load_ir152, compute_ir152_embedding

ATTACKERS = ['Facenet512', 'ArcFace', 'GhostFaceNet', 'VGG-Face']
VICTIMS = ['Facenet512', 'ArcFace', 'GhostFaceNet', 'VGG-Face', 'IR152']

def equivalent_models(a, b):
    na = str(a).strip().lower().replace('-', '').replace('_', '')
    nb = str(b).strip().lower().replace('-', '').replace('_', '')
    return na == nb

def success(sim, threshold, attack_type):
    return int(sim >= threshold) if str(attack_type).strip().lower() == 'impersonation_attack' else int(sim < threshold)

def impact(clean, adv, attack_type):
    return (adv - clean) if str(attack_type).strip().lower() == 'impersonation_attack' else (clean - adv)

def build_leaderboard(out_dir, base_name):
    # Base baseline summary
    base_file = f'results_baseline/subset_{base_name}.csv'
    if not os.path.exists(base_file):
        print(f"Skipping leaderboard for {base_name} as baseline CSV not found.")
        return
    dfs = [pd.read_csv(base_file)]
    
    # Student summaries
    for p in glob.glob(f'results_student_attacks/*/*_{base_name}.csv'):
        dfs.append(pd.read_csv(p))
    
    # PGN summary
    pgn_file = out_dir / f'pgn_{base_name}.csv'
    if os.path.exists(pgn_file):
        dfs.append(pd.read_csv(pgn_file))
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Sort
    if 'attacker_model' in combined.columns:
        combined = combined.sort_values(['attacker_model', 'victim_model', 'breach_rate_pct'], ascending=[True, True, False])
    elif 'attack_type' in combined.columns:
        combined = combined.sort_values(['attack_type', 'breach_rate_pct'], ascending=[True, False])
    else:
        combined = combined.sort_values('breach_rate_pct', ascending=False)
        
    combined.to_csv(out_dir / f'full_leaderboard_{base_name}.csv', index=False)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke-test', action='store_true', help='Run 2 pairs for 1 attacker')
    args = ap.parse_args()

    df = pd.read_csv('docs/subset_input_pairs.csv')
    if args.smoke_test:
        df = df.head(2)
        global ATTACKERS
        ATTACKERS = [ATTACKERS[0]]

    dataset_root = 'dataset_extractedfaces'
    out_dir = Path('results_pgn')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open('core/verification_thresholds.json') as f:
        thresholds = json.load(f)
    
    # Pre-load clean similarities
    raw_df = pd.read_csv('results_baseline/subset_raw_similarities_long.csv')
    clean_sims = raw_df[raw_df['attack_method'] == 'clean'].copy()
    
    raw_rows = []
    error_rows = []
    error_log = out_dir / 'errors.csv'
    
    # Cache victim models to avoid reloading
    victim_cache = {}
    def get_victim(v_name):
        if v_name not in victim_cache:
            if v_name == 'IR152':
                victim_cache[v_name] = load_ir152('core/ir152.pth')
            else:
                victim_cache[v_name] = build_attacker(v_name)
        return victim_cache[v_name]

    print("Starting PGN Generation & Evaluation Pipeline")
    for attacker in ATTACKERS:
        print(f"\n--- Loading Attacker Model: {attacker} ---")
        start_time = time.time()
        model = build_attacker(attacker)
        input_size = ATTACKER_MODELS[attacker]
        
        for _, rec in df.iterrows():
            row_id = int(rec['row_id'])
            try:
                src_path = resolve_image_path(rec['img1'], dataset_root)
                tgt_path = resolve_image_path(rec['img2'], dataset_root)
                
                src = tf.expand_dims(load_and_preprocess(src_path, input_size), 0)
                tgt = tf.expand_dims(load_and_preprocess(tgt_path, input_size), 0)
                
                # Generate PGN attack
                adv = run_attack('PGN', model, src, tgt, rec['attack_type'], input_size)
                adv_np = denormalize(adv.numpy()[0])
                
                # Save the adv image
                pgn_path = save_adv(adv_np, 'PGN', src_path, tgt_path, rec['attack_type'], attacker, row_id, str(out_dir))
                
                # Evaluate against victims
                for victim in VICTIMS:
                    if equivalent_models(attacker, victim):
                        continue
                    if equivalent_models(attacker, 'Facenet512') and equivalent_models(victim, 'Facenet'):
                        continue
                    
                    v_model = get_victim(victim)
                    if v_model is None:
                        continue # skipped IR152 if not found
                    
                    v_size = ATTACKER_MODELS.get(victim, (112, 112))
                    if victim == 'Facenet':
                        v_size = (160, 160)
                    elif victim == 'Facenet512':
                        v_size = (160, 160)
                    elif victim == 'ArcFace':
                        v_size = (112, 112)
                    elif victim == 'GhostFaceNet':
                        v_size = (112, 112)
                    elif victim == 'VGG-Face':
                        v_size = (224, 224)
                    
                    # Resize adv and tgt for victim
                    v_adv = tf.image.resize(adv, v_size)
                    v_tgt = tf.expand_dims(load_and_preprocess(tgt_path, v_size), 0) # re-load properly
                    
                    # We can just load clean images dynamically for victim size
                    v_tgt_img = tf.expand_dims(load_and_preprocess(tgt_path, v_size), 0)
                    
                    if victim == 'IR152':
                        # Use PyTorch pipeline for IR152
                        emb_adv = compute_ir152_embedding(v_model, v_adv)
                        emb_tgt = compute_ir152_embedding(v_model, v_tgt_img)
                        # np sum instead of tf reduce_sum because emb is numpy
                        import numpy as np
                        sim = float(np.sum(emb_adv * emb_tgt, axis=1)[0])
                    else:
                        emb_adv = compute_embedding(v_model, v_adv)
                        emb_tgt = compute_embedding(v_model, v_tgt_img)
                        sim = float(tf.reduce_sum(emb_adv * emb_tgt, axis=1).numpy()[0])
                    
                    raw_rows.append({
                        'row_id': row_id,
                        'attacker_model': attacker,
                        'img1': rec['img1'],
                        'img2': rec['img2'],
                        'dataset': rec['dataset'],
                        'attack_type': rec['attack_type'],
                        'source_csv': 'docs/subset_input_pairs.csv',
                        'victim_model': victim,
                        'attack_method': 'PGN',
                        'variant': 'vanilla',
                        'similarity': sim,
                        'source_column': 'pgn_path'
                    })
            except Exception as e:
                print(f"Error on row {row_id} with attacker {attacker}: {e}")
                error_rows.append({'row_id': row_id, 'attacker_model': attacker, 'error': str(e)})
                pd.DataFrame(error_rows).to_csv(error_log, index=False)
                
        # Checkpoint after each attacker model finishes
        if raw_rows:
            partial_df = pd.DataFrame(raw_rows)
            partial_df.to_csv(out_dir / f'pgn_subset_raw_similarities_partial.csv', index=False)
        
        elapsed = time.time() - start_time
        print(f"Completed {attacker} in {elapsed:.2f} seconds.")

    # Save raw similarities
    raw_out = pd.DataFrame(raw_rows)
    raw_out.to_csv(out_dir / 'subset_raw_similarities_long.csv', index=False)
    
    # Merge with clean for eval
    keep = pd.concat([clean_sims, raw_out], ignore_index=True)
    
    clean = keep[keep['attack_method'] == 'clean'][['row_id','attacker_model','victim_model','dataset','attack_type','similarity']].rename(columns={'similarity':'clean_similarity'})
    adv_df = keep[(keep['variant'] == 'vanilla') & (keep['attack_method'] == 'PGN')].copy()
    
    merged = adv_df.merge(clean, on=['row_id','attacker_model','victim_model','dataset','attack_type'], how='left')
    merged['threshold'] = merged.apply(lambda r: thresholds[r['victim_model']][r['dataset']]['threshold'], axis=1)
    merged['breach'] = merged.apply(lambda r: success(r['similarity'], r['threshold'], r['attack_type']), axis=1)
    merged['impact'] = merged.apply(lambda r: impact(r['clean_similarity'], r['similarity'], r['attack_type']), axis=1)
    merged = merged.rename(columns={'similarity':'adv_similarity'})
    merged.to_csv(out_dir / 'pgn_attack_eval_long.csv', index=False)
    
    # Generate Summaries
    merged.groupby('attack_method').agg(num_rows=('breach','size'), breach_rate_pct=('breach', lambda s: 100.0 * s.mean()), impact_mean=('impact','mean')).reset_index().sort_values('breach_rate_pct', ascending=False).to_csv(out_dir / 'pgn_attack_summary.csv', index=False)
    merged.groupby(['attack_type','attack_method']).agg(num_rows=('breach','size'), breach_rate_pct=('breach', lambda s: 100.0 * s.mean()), impact_mean=('impact','mean')).reset_index().sort_values(['attack_type','breach_rate_pct'], ascending=[True,False]).to_csv(out_dir / 'pgn_attack_summary_by_goal.csv', index=False)
    merged.groupby(['attacker_model','victim_model','attack_method']).agg(num_rows=('breach','size'), breach_rate_pct=('breach', lambda s: 100.0 * s.mean()), impact_mean=('impact','mean')).reset_index().sort_values(['attacker_model','victim_model','breach_rate_pct'], ascending=[True,True,False]).to_csv(out_dir / 'pgn_attacker_victim_summary.csv', index=False)

    print("\n--- Generating Leaderboards ---")
    build_leaderboard(out_dir, 'attack_summary')
    build_leaderboard(out_dir, 'attack_summary_by_goal')
    build_leaderboard(out_dir, 'attacker_victim_summary')
    print("Done. All files saved to results_pgn/")

if __name__ == '__main__':
    main()
