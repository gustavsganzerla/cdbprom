import os
from tqdm import tqdm
import multiprocessing

path = "path/to/your/input/data
path_out = "path/to/your/output/data

stability_values = {
    'AA': -1,
    'AT': -0.88,
    'TA': -0.58,
    'AG': -1.3,
    'GA': -1.3,
    'TT': -1,
    'AC': -1.45,
    'CA': -1.45,
    'TG': -1.44,
    'GT': -1.44,
    'TC': -1.28,
    'CT': -1.28,
    'CC': -1.84,
    'CG': -2.24,
    'GC': -2.27,
    'GG': -1.84
}

def process_file(filename):
    
    if filename.endswith('.ft'):
        
        with open(os.path.join(path, filename), 'r') as in_file:
            
            if not os.path.exists(path_out):
                os.makedirs(path_out)
            
            if os.access(path_out, os.W_OK):
                
                with open(os.path.join(path_out, filename.replace('.ft', '.txt')), 'w') as out_file:
                    
                    for line in in_file:
                        
                        cols = line.strip().split('\t')
                        
                        for x in cols:
                            if len(x)==400:
                                #this is applied for the sequences coming directly from the RSAT database
                                #we only selected the upstream region from 320 to +1 to convert.
                                out_file.write(cols[2] + '\t')
                                sequence = cols[6][320:400]
    
                                stability_list = [stability_values.get(sequence[i:i+2].upper(), 0) for i in range(0, len(sequence)-1)]
    
                                out_file.write('\t'.join(map(str, stability_list)) + '\n')
            else:
                print("Output directory is not accessible.")
filenames = os.listdir(path)

with multiprocessing.Pool() as pool:
    for _ in tqdm(pool.imap_unordered(process_file, filenames), total=len(filenames)):
        pass
