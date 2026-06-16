import os


def run_exp(phase, multigpu=False, cuda=0, bs=32, datasetname='SLAKE', llm='llama-2-7b', llmf='lora', padd='right',
            out_dir='checkpoints/baseline', testst1='experiments/baseline/pseudo'):
    if phase == 'train':
        if multigpu:
            the_command = ('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=' + str('0,1') \
                          + ' accelerate launch' \
                          + ' --config_file ' + 'default_config.yaml' \
                          + ' --main_process_port 29521' \
                          + ' --num_processes=' + str(2) \
                          + ' --num_machines=' + str(1) \
                          + ' --mixed_precision=' + 'fp16' \
                          + ' --deepspeed_config_file=' + 'zero2.json' \
                          + ' main.py' \
                          + ' --method=' + 'baseline' \
                          + ' --phase=' + phase \
                          + ' --dataset=' + datasetname \
                          + ' --batch_size=' + str(bs) \
                          + ' --pad=' + padd \
                          + ' --model_type=' + llm \
                          + ' --llmlora=' + llmf \
                          + ' --vislora=' + 'lora' \
                          + ' --lr=' + str(2e-5) \
                          + ' --out_dir=' + out_dir \
                          + ' --testst1=' + testst1)
        else:
            the_command = ('CUDA_VISIBLE_DEVICES=' + str(cuda)
                           + ' python main.py'
                           + ' --device=cuda:' + str(cuda) \
                           + ' --method=' + 'baseline' \
                           + ' --phase=' + phase \
                           + ' --dataset=' + datasetname \
                           + ' --batch_size=' + str(bs) \
                           + ' --pad=' + padd \
                           + ' --model_type=' + llm \
                           + ' --llmlora=' + llmf \
                           + ' --vislora=' + 'lora' \
                           + ' --lr=' + str(2e-5) \
                           + ' --out_dir=' + out_dir \
                           + ' --testst1=' + testst1
                           )
    elif phase == 'test':
        if multigpu:
            the_command = ('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=' + str('0,1') \
                          + ' accelerate launch' \
                          + ' --config_file ' + 'default_eval.yaml' \
                          + ' --main_process_port 29522' \
                          + ' --num_processes=' + str(2) \
                          + ' --num_machines=' + str(1) \
                          + ' --mixed_precision=' + 'fp16' \
                          + ' main.py' \
                          + ' --method=' + 'baseline' \
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
                          + ' --testst1=' + testst1)
        else:
            the_command = ('python main.py'
                           + ' --method=' + 'baseline' \
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
                           + ' --testst1=' + testst1
                           )

    os.system(the_command)


run_exp('train', multigpu=True, bs=32, datasetname='SLAKE', llm='llama-2-7b', llmf='lora', padd='right',
        out_dir='checkpoints/baseline', testst1='experiments/baseline')
run_exp('test', multigpu=True, bs=32, datasetname='SLAKE', llm='llama-2-7b', llmf='lora', padd='right',
        out_dir='checkpoints/baseline', testst1='experiments/baseline')