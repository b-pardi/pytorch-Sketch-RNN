import sys
import inspect
import os
import argparse
import torch
import numpy as np
import io

from sketch_rnn.hparams import hparam_parser
from train_sketch_rnn import train_sketch_rnn
from sketch_rnn.utils.misc import to_tensor
from infer import generate

PROJECT_DIR = os.path.abspath(".")
LOG_FILE_PATH = os.path.join(PROJECT_DIR, "trace.log")
LOG_DIR_SPLIT = "cloned\\"
frame_depth = {}
log_file = open(LOG_FILE_PATH, "a")

# Set numpy print options to ensure arrays are truncated
np.set_printoptions(edgeitems=3, threshold=30)

# Variables to track `<listcomp>` behavior
active_listcomp_frame = None
listcomp_function_calls = {}  # key: (filename, func_name, firstlineno), value: (count, skipped)

MAX_DETAILED_CALLS = 3

def format_value(value):
    """Format argument and return values for easier reading and safety."""
    if isinstance(value, torch.Tensor):
        # Brief summary for tensors
        out = f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    elif isinstance(value, io.IOBase):
        # File-like object
        out = f"<File object {getattr(value, 'name', '')}>"
    else:
        # Try a safe repr
        try:
            s = repr(value)
        except Exception:
            # If repr fails, fallback to simple type name
            out = f"<Unprintable {type(value).__name__}>"
        else:
            # Truncate if too long
            if len(s) > 500:
                s = s[:500] + "... (truncated)"

            # Remove null chars and non-printable chars
            s = s.replace('\x00', '\\x00')

            # Ensure ASCII
            s = s.encode('ascii', 'replace').decode('ascii')
            out = s

    return out

def format_args(args_dict):
    formatted = []
    for k, v in args_dict.items():
        formatted.append(f"{k}={format_value(v)}")
    return ", ".join(formatted)

def summarize_listcomp():
    """After exiting a <listcomp>, summarize any skipped calls."""
    global listcomp_function_calls
    for (filename, func_name, lineno), (count, skipped) in listcomp_function_calls.items():
        if skipped > 0:
            log_file.write(
                f"    ... plus {skipped} more calls to {func_name} in {filename}:{lineno} not logged in detail.\n\n"
            )

    # Reset after summarizing
    listcomp_function_calls.clear()

def trace_returns(frame, event, arg, func_key, call_number):
    """Traces returns for functions called inside or outside <listcomp>."""
    global active_listcomp_frame, listcomp_function_calls

    if event == 'return':
        depth = frame_depth.get(frame, 0)
        indent = " " * 4 * depth

        filename, func_name, lineno = func_key
        inside_listcomp = (active_listcomp_frame is not None)

        if func_name == '<listcomp>':
            # <listcomp> itself is never in listcomp_function_calls.
            # Just log return normally (or very briefly)
            # Since <listcomp> is not a project/torch function call we care to skip, we can just do:
            ret_val = arg
            log_file.write(f"{indent}RETURN: {format_value(ret_val)}\n\n")
        else:
            # For normal functions inside <listcomp>:
            if inside_listcomp and func_key in listcomp_function_calls:
                count, skipped = listcomp_function_calls[func_key]
                if count <= MAX_DETAILED_CALLS:
                    # Log return normally for first 3 calls
                    ret_val = arg
                    log_file.write(f"{indent}RETURN: {format_value(ret_val)}\n\n")
                # If count > 3, we do nothing (already counted as skipped)
            else:
                # Normal logging outside <listcomp> or if not found
                ret_val = arg
                log_file.write(f"{indent}RETURN: {format_value(ret_val)}\n\n")

        if frame in frame_depth:
            del frame_depth[frame]

        return None
    elif event == 'call':
        # We do not trace deeper calls from here
        return None

    return None

def trace_func(frame, event, arg):
    global active_listcomp_frame, listcomp_function_calls

    if event == 'call':
        code = frame.f_code
        filename = code.co_filename
        func_name = code.co_name
        lineno = code.co_firstlineno

        # If your environment is in project dir, exclude venv
        if 'venv' in filename:
            return None

        # Check if we’re entering a <listcomp>
        if func_name == '<listcomp>' and active_listcomp_frame is None:
            active_listcomp_frame = frame
            listcomp_function_calls = {}

        inside_listcomp = (active_listcomp_frame is not None)

        # Check if the call originates in the project directory or PyTorch
        if (filename.startswith(PROJECT_DIR) or 'torch' in filename) and 'venv' not in filename:
            # Determine depth
            parent_frame = frame.f_back
            depth = frame_depth.get(parent_frame, 0) + 1
            frame_depth[frame] = depth
            indent = " " * 4 * depth

            # Retrieve arguments
            args_info = inspect.getargvalues(frame)
            args_dict = {arg_name: args_info.locals[arg_name] for arg_name in args_info.args}

            func_key = (filename, func_name, lineno)

            if inside_listcomp and func_name != '<listcomp>':
                if func_key not in listcomp_function_calls:
                    listcomp_function_calls[func_key] = [0, 0]  # [count, skipped]

                listcomp_function_calls[func_key][0] += 1
                count, skipped = listcomp_function_calls[func_key]

                if count <= MAX_DETAILED_CALLS:
                    # Log normally for first 3 calls
                    log_file.write(f"{indent}CALL: {func_name} in {filename.split(LOG_DIR_SPLIT)[-1]}:{lineno}\n")
                    log_file.write(f"{indent}Arguments: {format_args(args_dict)}\n\n")
                    return lambda f, e, a: trace_returns(f, e, a, func_key, count)
                else:
                    # Increment skipped calls
                    listcomp_function_calls[func_key][1] += 1
                    # No detailed logging for these calls
                    return lambda f, e, a: trace_returns(f, e, a, func_key, count)
            else:
                # Normal logging outside <listcomp> or for <listcomp> itself
                log_file.write(f"{indent}CALL: {func_name} in {filename.split(LOG_DIR_SPLIT)}:{lineno}\n")
                log_file.write(f"{indent}Arguments: {format_args(args_dict)}\n\n")
                return lambda f, e, a: trace_returns(f, e, a, func_key, 1)
        else:
            # Not in project or torch, don't trace
            return None

    elif event == 'return':
        # If we're returning from a <listcomp>, summarize skipped calls
        if frame is active_listcomp_frame:
            summarize_listcomp()
            active_listcomp_frame = None

    return None

if __name__ == '__main__':
    sys.settrace(trace_func)
    
    hp_parser = hparam_parser()
    parser = argparse.ArgumentParser(parents=[hp_parser])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    try:
        #train_sketch_rnn(args)
        generate()
        #to_tensor(torch.tensor([1,2,3]))
        print("TRAINING DONE: Saving log file...")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"***** ERROR: training failed unexpectedly:\n{e}")
    finally:

        sys.settrace(None)
        log_file.flush()
        log_file.close()