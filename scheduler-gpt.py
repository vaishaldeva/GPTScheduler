#!/usr/bin/env python3
# scheduler-gpt.py
#
# Team Members:
# Matthew Salamone, Lucas Sutton, Justin Moreira, Vaishal Devasenapathy

import sys
import os
from collections import deque

# --------------------------------------------------------------------------
# --- Data Structures and Utilities
# --------------------------------------------------------------------------

class Process:
    """
    Represents a single process in the simulation.

    Attributes:
        name (str): The unique identifier for the process.
        arrival_time (int): The time tick at which the process arrives.
        burst_time (int): The total CPU time required by the process.
        tickets (int): The number of tickets for Stride Scheduling.
        priority (int): The priority level for Priority Scheduling (lower number is higher priority).
        
        remaining_burst (int): CPU time left for the process to complete.
        start_time (int): The time tick when the process first runs.
        finish_time (int): The time tick when the process completes.
        
        stride (float): The stride value for Stride Scheduling (10000 / tickets).
        pass_value (float): The pass value for Stride Scheduling.
        
        wait_time (int): Total time spent waiting in the ready queue.
        turnaround_time (int): Total time from arrival to completion.
        response_time (int): Time from arrival to first execution.
    """
    def __init__(self, name, arrival_time, burst_time, tickets=0, priority=0):
        self.name = name
        self.arrival_time = int(arrival_time)
        self.burst_time = int(burst_time)
        self.tickets = int(tickets)
        self.priority = int(priority)

        # Simulation state variables
        self.remaining_burst = self.burst_time
        self.start_time = -1
        self.finish_time = -1

        # Stride Scheduling specific attributes
        self.stride = 10000 / self.tickets if self.tickets > 0 else 0
        self.pass_value = 0

        # Final performance metrics
        self.wait_time = 0
        self.turnaround_time = 0
        self.response_time = -1

    def __repr__(self):
        """Developer-friendly representation for debugging."""
        return (f"Process(name='{self.name}', arrival={self.arrival_time}, "
                f"burst={self.burst_time}, rem={self.remaining_burst})")

def get_algorithm_display_name(use_code):
    """Translates the internal algorithm code to a user-friendly string."""
    return {
        'fcfs': "First-Come First-Served",
        'sjf': "preemptive Shortest Job First",
        'rr': "Round-Robin",
        'stride': "Stride Scheduling",
        'priority': "preemptive Priority Scheduling"
    }.get(use_code, "Unknown Algorithm")

# --------------------------------------------------------------------------
# --- Input Parsing and Validation
# --------------------------------------------------------------------------

def parse_input_file(filename):
    """
    Reads and validates the input file, creating Process objects.
    """
    config = {}
    processes = []
    required_params = {'processcount', 'runfor', 'use'}

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found.")
        sys.exit(1)

    # First pass to find config, especially the 'use' directive
    for line in lines:
        cleaned = line.strip().split('#', 1)[0].strip()
        if not cleaned: continue
        parts = cleaned.split()
        directive = parts[0].lower()
        if directive in required_params or directive == 'quantum':
            try:
                config[directive] = parts[1].lower() if directive == 'use' else int(parts[1])
            except (IndexError, ValueError):
                print(f"Error: Malformed configuration line: '{line.strip()}'")
                sys.exit(1)

    # Validate required configuration is present
    found_params = set(config.keys())
    if not required_params.issubset(found_params):
        missing = sorted(list(required_params - found_params))[0]
        print(f"Error: Missing parameter {missing}.")
        sys.exit(1)

    if config['use'] == 'rr' and 'quantum' not in config:
        print("Error: Missing quantum parameter when use is 'rr'")
        sys.exit(1)
        
    # Second pass to parse processes based on the algorithm
    for line_num, line in enumerate(lines, 1):
        cleaned = line.strip().split('#', 1)[0].strip()
        if not cleaned or cleaned.split()[0].lower() != 'process':
            continue
        
        parts = cleaned.split()
        try:
            p_info = {parts[i]: parts[i+1] for i in range(1, len(parts), 2)}
            proc = Process(
                name=p_info['name'],
                arrival_time=p_info['arrival'],
                burst_time=p_info['burst'],
                tickets=p_info.get('tickets', 0),
                priority=p_info.get('priority', 0)
            )
            if config['use'] == 'stride' and 'tickets' not in p_info:
                print(f"Error: Process {p_info['name']} requires 'tickets' for Stride Scheduling.")
                sys.exit(1)
            if config['use'] == 'priority' and 'priority' not in p_info:
                print(f"Error: Process {p_info['name']} requires 'priority' for Priority Scheduling.")
                sys.exit(1)
            processes.append(proc)
        except (KeyError, IndexError):
            print(f"Error: Malformed process line {line_num}: '{line.strip()}'")
            sys.exit(1)

    return config, processes

# --------------------------------------------------------------------------
# --- Simulation Core Logic
# --------------------------------------------------------------------------

def run_simulation(config, processes, out_file):
    """
    Main simulation loop dispatcher.
    """
    # Processes that need to be considered for scheduling are sorted by arrival time.
    incoming_processes = sorted(processes, key=lambda p: p.arrival_time)
    
    simulation_functions = {
        'fcfs': run_fcfs,
        'sjf': run_sjf,
        'rr': run_rr,
        'stride': run_stride,
        'priority': run_priority
    }
    
    # Get the appropriate simulation function for the selected algorithm
    sim_func = simulation_functions.get(config['use'])
    if not sim_func:
        print(f"Error: Unknown algorithm '{config['use']}' specified.")
        sys.exit(1)
        
    # Write the standard header
    out_file.write(f"{config['processcount']} processes\n")
    out_file.write(f"Using {get_algorithm_display_name(config['use'])}\n")
    if config['use'] == 'rr':
        out_file.write(f"Quantum   {config['quantum']}\n")
    out_file.write("\n")

    # Run the simulation
    sim_func(config, incoming_processes, out_file)

    # Write the final summary and metrics
    write_final_metrics(out_file, processes, config['runfor'])

def run_fcfs(config, incoming_procs, out_file):
    time, currently_running = 0, None
    ready_queue = deque()
    
    while time < config['runfor']:
        check_for_arrivals(time, incoming_procs, ready_queue, out_file)

        if currently_running and currently_running.remaining_burst == 0:
            handle_completion(time, currently_running, out_file)
            currently_running = None

        if not currently_running and ready_queue:
            currently_running = ready_queue.popleft()
            select_process(time, currently_running, out_file)

        log_tick(time, currently_running, out_file)
        time += 1

def run_sjf(config, incoming_procs, out_file):
    time, currently_running = 0, None
    ready_queue = []
    
    while time < config['runfor']:
        check_for_arrivals(time, incoming_procs, ready_queue, out_file)

        if currently_running and currently_running.remaining_burst == 0:
            handle_completion(time, currently_running, out_file)
            currently_running = None

        # This is the core preemption logic for SJF.
        # At each tick, we must decide if the current process is still the shortest.
        if ready_queue:
            # Sort the ready queue to find the absolute shortest job available.
            ready_queue.sort(key=lambda p: (p.remaining_burst, p.arrival_time))
            shortest_in_queue = ready_queue[0]

            if currently_running is None or \
               shortest_in_queue.remaining_burst < currently_running.remaining_burst:
                # If a shorter job is in the queue, preempt the current one.
                if currently_running:
                    ready_queue.append(currently_running)
                currently_running = ready_queue.pop(0)
                select_process(time, currently_running, out_file)

        log_tick(time, currently_running, out_file)
        time += 1

def run_rr(config, incoming_procs, out_file):
    time, currently_running = 0, None
    ready_queue = deque()
    quantum = config['quantum']
    time_slice_remaining = 0
    
    while time < config['runfor']:
        check_for_arrivals(time, incoming_procs, ready_queue, out_file)

        # A process must be checked for completion *before* its quantum expires.
        if currently_running:
            if currently_running.remaining_burst == 0:
                handle_completion(time, currently_running, out_file)
                currently_running = None
                time_slice_remaining = 0  # Reset slice
            elif time_slice_remaining == 0:
                # Quantum expired, move process to the end of the queue.
                ready_queue.append(currently_running)
                currently_running = None

        if not currently_running and ready_queue:
            currently_running = ready_queue.popleft()
            time_slice_remaining = quantum  # Start new time slice
            select_process(time, currently_running, out_file)

        log_tick(time, currently_running, out_file, decrement_slice=True)
        if currently_running:
             time_slice_remaining -= 1
        time += 1
        
def run_stride(config, incoming_procs, out_file):
    time, currently_running = 0, None
    ready_queue = []

    while time < config['runfor']:
        check_for_arrivals(time, incoming_procs, ready_queue, out_file)

        if currently_running and currently_running.remaining_burst == 0:
            handle_completion(time, currently_running, out_file)
            currently_running = None
        
        # The running process must re-join the ready queue to be considered
        # for the next time tick against any other ready processes.
        if currently_running:
            ready_queue.append(currently_running)
            currently_running = None

        if ready_queue:
            # Select process with the lowest pass value, tie-breaking with arrival time.
            ready_queue.sort(key=lambda p: (p.pass_value, p.arrival_time))
            currently_running = ready_queue.pop(0)
            select_process(time, currently_running, out_file)

            # Update its pass value for the next cycle.
            currently_running.pass_value += currently_running.stride

        log_tick(time, currently_running, out_file)
        time += 1

def run_priority(config, incoming_procs, out_file):
    """Preemptive Priority Scheduling. Lower number = higher priority."""
    time, currently_running = 0, None
    ready_queue = []
    
    while time < config['runfor']:
        check_for_arrivals(time, incoming_procs, ready_queue, out_file)

        if currently_running and currently_running.remaining_burst == 0:
            handle_completion(time, currently_running, out_file)
            currently_running = None

        # This is the core preemption logic for Priority Scheduling.
        # At each tick, we must decide if the current process still has the highest priority.
        if ready_queue:
            # Sort the ready queue to find the process with the highest priority (lowest value).
            ready_queue.sort(key=lambda p: (p.priority, p.arrival_time))
            highest_priority_in_queue = ready_queue[0]

            if currently_running is None or \
               highest_priority_in_queue.priority < currently_running.priority:
                # If a higher-priority job is in the queue, preempt the current one.
                if currently_running:
                    ready_queue.append(currently_running)
                currently_running = ready_queue.pop(0)
                select_process(time, currently_running, out_file)

        log_tick(time, currently_running, out_file)
        time += 1

# --------------------------------------------------------------------------
# --- Simulation Helper Functions
# --------------------------------------------------------------------------

def check_for_arrivals(time, incoming_procs, ready_queue, out_file):
    """Adds processes that have arrived at the current time to the ready queue."""
    while incoming_procs and incoming_procs[0].arrival_time == time:
        process = incoming_procs.pop(0)
        ready_queue.append(process)
        out_file.write(f"Time {time:3} : {process.name} arrived\n")

def handle_completion(time, process, out_file):
    """Sets finish time and logs the completion event."""
    process.finish_time = time
    out_file.write(f"Time {time:3} : {process.name} finished\n")

def select_process(time, process, out_file):
    """Sets response time (if not set) and logs the selection event."""
    if process.response_time == -1:
        process.response_time = time - process.arrival_time
    out_file.write(f"Time {time:3} : {process.name} selected (burst {process.remaining_burst:3})\n")

def log_tick(time, running_process, out_file, decrement_slice=False):
    """Decrements burst time for the running process or logs Idle."""
    if running_process:
        running_process.remaining_burst -= 1
    else:
        out_file.write(f"Time {time:3} : Idle\n")

def write_final_metrics(out_file, processes, simulation_time_limit):
    """Calculates and writes the final performance metrics for all processes."""
    out_file.write(f"Finished at time {simulation_time_limit}\n\n")

    # Sort processes by name for consistent output order
    processes.sort(key=lambda p: p.name)

    for p in processes:
        if p.finish_time != -1 and p.finish_time <= simulation_time_limit:
            p.turnaround_time = p.finish_time - p.arrival_time
            p.wait_time = p.turnaround_time - p.burst_time
            out_file.write(f"{p.name} wait {p.wait_time} turnaround {p.turnaround_time} response {p.response_time}\n")
        else:
            out_file.write(f"{p.name} did not finish\n")

# --------------------------------------------------------------------------
# --- Main Execution
# --------------------------------------------------------------------------

def main():
    """
    Entry point of the script. Handles command-line arguments,
    parsing, and orchestrates the simulation.
    """
    if len(sys.argv) != 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} <input file>")
        sys.exit(1)

    input_filename = sys.argv[1]
    config, processes = parse_input_file(input_filename)
    
    # CORRECTION: The output file path should be based on the basename of the
    # input file, not its full path. This ensures the output file is created
    # in the current working directory, which is what the test bench expects.
    base_name_with_ext = os.path.basename(input_filename)
    base_name = os.path.splitext(base_name_with_ext)[0]
    output_filename = f"{base_name}.out"

    try:
        with open(output_filename, 'w') as out_file:
            run_simulation(config, processes, out_file)
        # This print statement is for interactive use, not strictly needed for the test bench
        # print(f"Simulation complete. Output written to '{output_filename}'.")
    except IOError as e:
        print(f"Error: Could not write to output file '{output_filename}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
