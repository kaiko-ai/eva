import concurrent.futures
import tkinter as tk

def run_script(script_number):
    # Replace the following line with the actual command you want to run
    command = f"python script{script_number}.py"
    print(f"Running script {script_number}: {command}")
    # You can use subprocess or other methods to run the command here

def run_scripts_parallel(num_runs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_runs) as executor:
        executor.map(run_script, range(1, num_runs + 1))

def on_run_button_click(num_runs_entry):
    num_runs = int(num_runs_entry.get())
    run_scripts_parallel(num_runs)

# Simple tkinter GUI
def create_ui():
    root = tk.Tk()
    root.title("Parallel Script Runner")

    label = tk.Label(root, text="Number of runs:")
    label.pack()

    num_runs_entry = tk.Entry(root)
    num_runs_entry.pack()

    run_button = tk.Button(root, text="Run Scripts", command=lambda: on_run_button_click(num_runs_entry))
    run_button.pack()

    root.mainloop()

if __name__ == "__main__":
    create_ui()
