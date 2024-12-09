import cProfile
import pstats
import subprocess
import os
import sys
from sketch_rnn.hparams import hparam_parser
import argparse
import cairosvg
import pstats
from train_sketch_rnn import train_sketch_rnn


def svg2png(svg_file, png_file, dpi=300):
    cairosvg.svg2png(url=svg_file, write_to=png_file, dpi=dpi)
    print(f"High-resolution PNG saved to {png_file}")

def generate_callgraph(prof_file, dot_file, output_file, threshold, output_format='png'):
    # Generate a DOT file using gprof2dot
    subprocess.run([
        'gprof2dot',
        prof_file, 
        '--strip', 
        '-f', 'pstats',
        '-e', str(threshold),
        '-n', str(threshold),
        '-o', dot_file
    ], check=True)

    # Convert the DOT file to the desired format using Graphviz
    subprocess.run([
        'dot', f'-T{output_format}', dot_file, '-o', output_file
    ], check=True)

    print(f"Call graph saved to {output_file}")

def generate_flamegraph(prof_file, flamegraph_file):
    # Generate flamegraph using flameprof
    with open(flamegraph_file, 'w') as flame_output:
        subprocess.run(
            ['flameprof', prof_file],
            stdout=flame_output,
            check=True
        )
    print(f"Flamegraph saved to {flamegraph_file}")

if __name__ == '__main__':
    os.makedirs('profiles/', exist_ok=True)
    prof_file = 'profiles/output.prof'
    dot_file = 'profiles/callgraph.dot'
    callgraph_file = 'profiles/callgraph.png'
    flamegraph_file = 'profiles/flamegraph.svg'

    # Profile the main script
    profiler = cProfile.Profile()
    profiler.enable()

    hp_parser = hparam_parser()
    parser = argparse.ArgumentParser(parents=[hp_parser])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    try:
        train_sketch_rnn(args)
        print("ETSTSETSETSETse")

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"***** ERROR: training failed unexpectedly:\n{e}")
    finally:
        profiler.disable()
        # Save profiling data
        profiler.dump_stats(prof_file)
        print(f"Profiling data saved to {prof_file}")

        # Generate the call graph
        try:
            generate_callgraph(prof_file, dot_file, 'profiles/callgraph.svg', 0.001, 'svg')
            svg2png('profiles/callgraph.svg', 'profiles/callgraph.png')
        except Exception as e:
            print(f"Error generating call graph: {e}")
