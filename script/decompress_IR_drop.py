import os

decompress_path = '../IR_drop_features_decompressed'
os.system('mkdir -p %s ' % (decompress_path))
os.system('cat ./Ir_drop_features/power_t.tar.gz.* > ./Ir_drop_features/power_t.tar.gz')

filelist = os.walk('../IR_drop_features')
for parent,dirnames,filenames in filelist:
    for filename in filenames:
        if os.path.splitext(filename)[1] == '.gz':
            filepath = os.path.join(parent, filename)
            os.system('gzip -dk %s' % filepath)
            os.system('tar -xf %s -C %s' % (filepath.replace('.gz',''), parent.replace('IR_drop_features','IR_drop_features_decompressed')))
