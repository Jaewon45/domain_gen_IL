#!/usr/bin/env python3
"""One source-only IRM/VREx calibration candidate from an ERM warm-up."""
from __future__ import annotations
import argparse, copy, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..algorithms import DomainAlgorithm
from ..data import InfiniteDomainBatches, datasets_from_manifest, load_imagenet100, proportional_batch_sizes
from ..manifests import load_manifest
from ..models import build_model, build_transform_and_spec


def evaluate_source_domains(model, datasets, device, batch_size, workers):
    """Source-held-out CE/accuracy plus prediction-collapse diagnostics."""
    rows = []
    model.eval()
    with torch.no_grad():
        for domain, dataset in datasets.items():
            kwargs = dict(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
            if workers > 0:
                kwargs.update(persistent_workers=True, prefetch_factor=2)
            loader = DataLoader(**kwargs)
            count = correct = 0; ce_sum = entropy_sum = 0.0; predicted = torch.zeros(100, dtype=torch.long)
            for images, targets in loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                logits = model(images, torch.zeros((len(images), 1), device=device))
                probabilities, prediction = logits.softmax(dim=1), logits.argmax(dim=1)
                ce_sum += float(F.cross_entropy(logits, targets, reduction='sum'))
                entropy_sum += float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)).sum())
                correct += int((prediction == targets).sum()); count += int(targets.numel())
                predicted += torch.bincount(prediction.cpu(), minlength=100)
            rows.append({'domain': domain, 'count': count, 'cross_entropy': ce_sum / count,
                         'accuracy': correct / count, 'prediction_entropy': entropy_sum / count,
                         'largest_predicted_class_fraction': int(predicted.max()) / count})
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--method', choices=('irm', 'vrex', 'groupdro'), required=True); p.add_argument('--penalty-weight',type=float,default=1.0)
    p.add_argument('--groupdro-eta',type=float,default=None)
    p.add_argument('--steps',type=int,default=100); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--num-lambda-samples',type=int,default=4)
    p.add_argument('--iro-sampler-learning-rate',type=float,default=1e-6)
    p.add_argument('--workers',type=int,default=1); p.add_argument('--batch-size',type=int,default=64)
    args=p.parse_args()
    if args.penalty_weight <= 0 or args.steps <= 0: raise ValueError('penalty-weight and steps must be positive')
    if args.method == 'groupdro' and (args.groupdro_eta is None or args.groupdro_eta <= 0): raise ValueError('groupdro requires a positive groupdro-eta')
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m=load_manifest(args.manifest)
    if m['condition'] != 'balanced' or int(m['global_seed']) != args.seed: raise ValueError('Calibration requires matching balanced manifest and seed')
    if set(m['active_domain_counts']) != set(m['source_validation_assignments']): raise ValueError('All balanced source domains need held-out validation')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=False)
    transform,_=build_transform_and_spec()
    train=load_imagenet100(m['dataset']['name'],m['dataset']['revision'],split='train')
    datasets=datasets_from_manifest(train,m,transform,validation=False)
    validation_datasets=datasets_from_manifest(train,m,transform,validation=True)
    batches=InfiniteDomainBatches(datasets,proportional_batch_sizes(m['active_domain_counts'],args.batch_size),args.seed,args.workers,pin_memory=True,persistent_workers=args.workers > 0,prefetch_factor=2)
    base=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=build_model(base['model_mode'], weight_version=base['weight_version']).to(device); model.load_state_dict(base['model_state'])
    alg=DomainAlgorithm(args.method,model,learning_rate=3e-4,weight_decay=1e-4,groupdro_eta=args.groupdro_eta or .1,num_lambda_samples=args.num_lambda_samples,eqrm_alpha=.9,penalty_weight=args.penalty_weight,erm_pretrain_iters=0,lr_cos_sched=False,lr_factor_reduction=1,total_steps=args.steps,seed=args.seed,device=device,iro_sampler_learning_rate=args.iro_sampler_learning_rate)
    records=[]
    for step in range(1,args.steps+1):
        update=alg.update(batches.next(step)); records.append({'method':args.method,'penalty_weight':args.penalty_weight,'step':step,**update})
    source_rows=evaluate_source_domains(model,validation_datasets,device,args.batch_size,args.workers)
    summary={'source_validation_mean_accuracy':float(np.mean([row['accuracy'] for row in source_rows])), 'source_validation_worst_accuracy':float(np.min([row['accuracy'] for row in source_rows])), 'source_validation_mean_cross_entropy':float(np.mean([row['cross_entropy'] for row in source_rows])), 'source_validation_mean_prediction_entropy':float(np.mean([row['prediction_entropy'] for row in source_rows])), 'source_validation_largest_predicted_class_fraction':float(max(row['largest_predicted_class_fraction'] for row in source_rows)), 'final_raw_penalty':records[-1].get('penalty'), 'final_domain_loss_spread':float(np.ptp(list(records[-1]['environment_losses'].values())))}
    provenance={'schema_version':1,'kind':'source_only_calibration','method':args.method,'seed':args.seed,'condition':'balanced','penalty_weight':args.penalty_weight,'groupdro_eta':args.groupdro_eta,'num_lambda_samples':args.num_lambda_samples,'iro_sampler_learning_rate':args.iro_sampler_learning_rate,'penalty_anneal_iters':400,'total_updates':400+args.steps,'warmup_updates':400,'main_updates':args.steps,'hyperparameter_selection':'source_validation_balanced','selection_rule':'maximize worst-source validation accuracy; break ties by mean source-validation accuracy; reject numerical instability or degenerate classifiers','checkpoint_selection':'final','checkpoint':str(Path(args.checkpoint).resolve()),'checkpoint_manifest':str(Path(args.manifest).resolve()),'checkpoint_manifest_id':m['manifest_id'],'target_or_clean_imagenet_loaded':False,'source_validation_domain_rows':source_rows,'training_summary':summary}
    torch.save({'method':args.method,'penalty_weight':args.penalty_weight,'model_state':copy.deepcopy(model.state_dict()),'provenance':provenance},out/'final.pt')
    with (out/'history.jsonl').open('w') as f:
      for r in records: f.write(json.dumps(r,sort_keys=True)+'\n')
    (out/'manifest.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output_dir':str(out),**summary},sort_keys=True))

if __name__=='__main__': main()
