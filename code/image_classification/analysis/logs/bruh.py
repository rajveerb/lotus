with open('code/image_classification/analysis/logs/imagenet_run.log','r') as f:
    lines = f.readlines()
    super_sum = 0
    for line in lines:
        if 'Elapsed Time: ' in line:
            time = float(line.split('Elapsed Time: ')[1][:-2])
            super_sum += time
    print(super_sum/3600)
            