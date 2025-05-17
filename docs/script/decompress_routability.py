import os

decompress_path = '../routability_features_decompressed'
os.system("mkdir -p %s " % (decompress_path))
filelist = os.walk('../routability_features')

for parent,dirnames,filenames in filelist:
    for filename in filenames:
        if os.path.splitext(filename)[1] == '.gz':
            filepath = os.path.join(parent, filename)
            os.system('gzip -dk %s' % filepath)
            os.system('tar -xf %s -C %s' % (filepath.replace('.gz',''), parent.replace('routability_features','routability_features_decompressed')))
