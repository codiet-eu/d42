from subprocess import Popen, PIPE
import os
import glob
import pandas as pd
import json

'''
Connector to R implementations.
@author Fadwa Idlahcen
'''
def clear_args(dir_path):
    files = glob.glob(dir_path + '/args/*')
    for f in files:
        os.remove(f)


def clear_results(dir_path):
    files = glob.glob(dir_path + '/results/*')
    for f in files:
        os.remove(f)

def run_R(script, arg_dict):

    args_json = json.dumps(arg_dict)

    # Check if the method is available
    available_scripts = ["bnstruct", "mcmc_bidag"] #TO DO:add dbnr as well
    if script not in available_scripts:
        print("Error: method not available")
        return None

    # Get the directory path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(dir_path, f"{script}.R")

    # Clear previous arguments and results
    clear_directory(os.path.join(dir_path, "args"))
    clear_directory(os.path.join(dir_path, "results"))

    cmd = ["Rscript", script_path, args_json]
    p = Popen(cmd, cwd="./", stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, error = p.communicate()

    print(output.decode())

    if p.returncode == 0:
        print('R Done')
        result_df = pd.read_csv(os.path.join(dir_path, "results", "result.csv"), header=0, index_col=0)
        return result_df
    else:
        print('R Error:\n {0}'.format(error.decode()))
        return None

def clear_directory(directory):
    # Clear the contents of a directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                clear_directory(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

