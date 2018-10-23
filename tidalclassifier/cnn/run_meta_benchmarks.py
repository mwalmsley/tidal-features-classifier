from tidalclassifier.utils.helper_funcs import from_json
from tidalclassifier.cnn.meta_benchmarks import benchmarkAllOnTables

name_to_benchmark = 'tb_m8' # TODO: pass as parameter?
instruct = from_json(name_to_benchmark + '_0_instruct.txt')
benchmarkAllOnTables(instruct)