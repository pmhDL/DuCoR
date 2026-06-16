import os
import shutil


def prepare_pseudo_answers(baseline_testst1, testst1, datasetname, llm):
    src_dir = os.path.join(baseline_testst1, datasetname, llm)
    dst_dir = os.path.join(testst1, datasetname, llm)
    src = os.path.join(src_dir, 'test_result.json')
    dst = os.path.join(dst_dir, 'test_result.json')
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, dst)


def prepare_warm_start(testst1, datasetname, llm, warm_start_pseudo_dir=None):
    if warm_start_pseudo_dir is None:
        return
    target_dir = os.path.join(testst1, datasetname, llm)
    os.makedirs(target_dir, exist_ok=True)
    for filename in ('gmm_epoch28.npz', 'protos_sigma_epoch28.pt'):
        src = os.path.join(warm_start_pseudo_dir, filename)
        dst = os.path.join(target_dir, filename)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)


def run_exp(phase, multigpu=False, cuda=0, bs=32, datasetname='SLAKE', llm='llama-2-7b', llmf='lora', padd='right',
            out_dir='checkpoints/baseline', testst1='experiments/ducor',
            baseline_testst1='experiments/baseline', warm_start_pseudo_dir=None):
    if phase == 'train':
        prepare_pseudo_answers(baseline_testst1, testst1, datasetname, llm)
        prepare_warm_start(testst1, datasetname, llm, warm_start_pseudo_dir)
        if multigpu:
            the_command = ('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=' + str('0,1') \
                          + ' accelerate launch' \
                          + ' --config_file ' + 'default_config.yaml' \
                          + ' --main_process_port 29523' \
                          + ' --num_processes=' + str(2) \
                          + ' --num_machines=' + str(1) \
                          + ' --mixed_precision=' + 'fp16' \
                          + ' --deepspeed_config_file=' + 'zero2.json' \
                          + ' main.py' \
                          + ' --method=' + 'ducor' \
                          + ' --phase=' + phase \
                          + ' --dataset=' + datasetname \
                          + ' --batch_size=' + str(bs) \
                          + ' --pad=' + padd \
                          + ' --model_type=' + llm \
                          + ' --llmlora=' + llmf \
                          + ' --vislora=' + 'lora' \
                          + ' --lr=' + str(2e-5) \
                          + ' --out_dir=' + out_dir \
                          + ' --testst1=' + testst1 \
                          + ' --ducor_include_val_in_train=' + 'no' \
                          + ' --ducor_save_strategy=' + 'val_acc')
        else:
            the_command = ('CUDA_VISIBLE_DEVICES=' + str(cuda)
                           + ' python main.py'
                           + ' --device=cuda:' + str(cuda) \
                           + ' --method=' + 'ducor' \
                           + ' --phase=' + phase \
                           + ' --dataset=' + datasetname \
                           + ' --batch_size=' + str(bs) \
                           + ' --pad=' + padd \
                           + ' --model_type=' + llm \
                           + ' --llmlora=' + llmf \
                           + ' --vislora=' + 'lora' \
                           + ' --lr=' + str(2e-5) \
                           + ' --out_dir=' + out_dir \
                           + ' --testst1=' + testst1 \
                           + ' --ducor_include_val_in_train=' + 'no' \
                           + ' --ducor_save_strategy=' + 'val_acc'
                           )
    elif phase == 'test':
        if multigpu:
            the_command = ('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=' + str('0,1') \
                          + ' accelerate launch' \
                          + ' --config_file ' + 'default_eval.yaml' \
                          + ' --main_process_port 29524' \
                          + ' --num_processes=' + str(2) \
                          + ' --num_machines=' + str(1) \
                          + ' --mixed_precision=' + 'fp16' \
                          + ' main.py' \
                          + ' --method=' + 'ducor' \
                          + ' --phase=' + phase \
                          + ' --dataset=' + datasetname \
                          + ' --batch_size=' + str(bs) \
                          + ' --pad=' + padd \
                          + ' --gen=' + 'bs' \
                          + ' --model_type=' + llm \
                          + ' --llmlora=' + llmf \
                          + ' --vislora=' + 'lora' \
                          + ' --lr=' + str(2e-5) \
                          + ' --out_dir=' + out_dir \
                          + ' --testst1=' + testst1 \
                          + ' --ducor_include_val_in_train=' + 'no' \
                          + ' --ducor_save_strategy=' + 'val_acc')
        else:
            the_command = ('python main.py'
                           + ' --method=' + 'ducor' \
                           + ' --phase=' + phase \
                           + ' --dataset=' + datasetname \
                           + ' --device=cuda:' + str(cuda) \
                           + ' --batch_size=' + str(bs) \
                           + ' --pad=' + padd \
                           + ' --gen=' + 'sig' \
                           + ' --model_type=' + llm \
                           + ' --llmlora=' + llmf \
                           + ' --vislora=' + 'lora' \
                           + ' --lr=' + str(2e-5) \
                           + ' --out_dir=' + out_dir \
                           + ' --testst1=' + testst1 \
                           + ' --ducor_include_val_in_train=' + 'no' \
                           + ' --ducor_save_strategy=' + 'val_acc'
                           )

    os.system(the_command)

# Run baseline first under the same out_dir root so ducor can load the baseline checkpoint.
# The baseline pseudo answers are copied from baseline_testst1 into testst1 before each train.

warm_start_pseudo_dir = None
baseline_testst1 = 'experiments/baseline'
ducor_testst1 = 'experiments/ducor'

run_exp('train', multigpu=False, datasetname='SLAKE', llm='llama-2-7b', llmf='lora', padd='right',
        out_dir='checkpoints/baseline', testst1=ducor_testst1, baseline_testst1=baseline_testst1, warm_start_pseudo_dir=warm_start_pseudo_dir)
run_exp('test', multigpu=False, datasetname='SLAKE', llm='llama-2-7b', llmf='lora', padd='right',
        out_dir='checkpoints/baseline', testst1=ducor_testst1)


