import glob
import logging
import os
import pathlib
import random
import json

import numpy as np
import pandas as pd
from pysat.formula import CNF
from pysat.examples.genhard import PHP
from pysat.solvers import Solver
import torch.optim.lr_scheduler as lr_scheduler

import gc
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, KLDivLoss, BCEWithLogitsLoss, L1Loss
from tqdm import tqdm
from utils.transformer import *

import flags
import model.circuit as circuit

from utils.latency import timer
from utils.baseline_sat import BaselineSolverRunner

class roundfunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        
        z = torch.where(x < 0.9, 0., 1.) 
        return z

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output
        
round = roundfunc.apply





class Runner(object):
    def __init__(self, problem_type: str = "sat"):
        self.args = flags.parse_args()
        self.problem_type = problem_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = self.args.batch_size
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        self.model = None
        self.loss = None
        self.optimizer = None
        self.module_name = None
        self.baseline = None
        self.save_dir = pathlib.Path(__file__).parent.parent / "results"
        self.datasets = []
        self.num_inputs = None
        self.dataset_str = ""
        self._setup_problems()

    def _setup_problems(self):
        """Setup the problem.
        Implementation will later be extended to support other problem types. (currnetly only SAT)
            - SAT: dataset will be either (1) .cnf files in args.dataset_path or (2) PySAT PHP problems.
        """
        if self.args.problem_type == "cnf":
            if self.args.dataset_path is None:
                raise NotImplementedError
                logging.info("No dataset found. Generating PySAT PHP problems.")
            else:
                dataset_path = os.path.join(pathlib.Path(__file__).parent.parent, self.args.dataset_path)
                self.datasets = sorted(glob.glob(dataset_path))
                self.problems = None
                self.dataset_str = self.args.dataset_path.split('/')[-1]
            self.results={}
            logging.info(f"Dataset used: {self.dataset_str}")
        else:
            raise NotImplementedError

    def read_SAT_file(self, file_path):
        return CNF(file_path)


    def _initialize_model(self, prob_id: int = 0):
        """Initialize problem-specifc model, loss, optimizer and input, target tensors
        for a given problem instance, e.g. a SAT problem (in CNF form).
        Note: must re-initialize the model for each problem instance.

        Args:
            prob_id (int, optional): Index of the problem to be solved. Defaults to 0.
        """
        problem = self.read_SAT_file(self.datasets[prob_id])

        if self.args.problem_type == "cnf":
            file_name = f"{self.dataset_str}.json"
            directory = "../models"  # Replace with your directory path

            # Check if the file exists
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                with open(file_path, "r") as json_file:
                    loaded_model_info = json.load(json_file)
                class_definition = loaded_model_info["model_script"]
                hyperparameters = loaded_model_info["hyperparameters"]
                self.num_inputs = hyperparameters["num_inputs"]
                self.num_outputs = hyperparameters["num_outputs"]
            else:
                inputs, outputs, variables, clause_gate, clause_output = transformer(problem)
                class_definition, extra_outputs = generate_pytorch_model(inputs, outputs, variables, clause_gate, clause_output)
                model_info = {
                        "model_script": class_definition,  # Replace with your model name
                        "hyperparameters": {
                        "num_inputs": len(inputs),
                        "num_outputs": len(outputs) + extra_outputs
                    }
                }
                with open(file_path, "w") as json_file:
                    json.dump(model_info, json_file)
                
                self.num_inputs = len(inputs)
                self.num_outputs = len(outputs) + extra_outputs

            
            
            

        self.model = circuit.CircuitModel(
                pytorch_model=class_definition,
                num_inputs=self.num_inputs,
                num_outputs= self.num_outputs,
                device = self.device,
                batch_size = self.batch_size
            )
        self.target = torch.ones(self.batch_size, self.num_outputs, device = self.device)
        self.loss = MSELoss(reduction='sum')
        
        
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.args.learning_rate
        )
        self.results[prob_id] = {
                "prob_desc": self.datasets[prob_id].split('/')[-1],
                "num_outputs": self.num_outputs,
                "num_inputs": self.num_inputs,
            }

        self.model.to(self.device)
        self.target.to(self.device)
        self.epochs_ran = 0



    def run_back_prop(self, train_loop: range):
        """Run backpropagation for the given number of epochs."""
        self.model.train()
        for epoch in train_loop:
            self.optimizer.zero_grad()
            outputs_list, _ = self.model()
            output = torch.cat(outputs_list , dim = -1)
            loss = self.loss((output), self.target) 
            loss.backward()
            self.optimizer.step()
        return None

    
    def _check_solution(self):

        with torch.no_grad():
            primary_ins = self.model.emb()
            output_list, variables_list = self.model.probabilistic_circuit_model([torch.round(term) for term in primary_ins])
        zero_indices = torch.nonzero(torch.cat(output_list,dim=-1).squeeze(0) == 0.0, as_tuple=False).squeeze()
        valid_solutions_idx = torch.cat(output_list,dim=-1).prod(dim = -1) > 0.5
        solutions = torch.unique( torch.cat(variables_list,dim=-1)[valid_solutions_idx,:], dim = 0)
        num_unique_solutions = len(solutions)

        return num_unique_solutions

    
    
    def run_model(self, prob_id: int = 0):
        solutions_found = []
        if self.args.latency_experiment:
            train_loop = range(self.args.num_steps)
            elapsed_time, _ = timer(self.run_back_prop)(train_loop)
            logging.info("--------------------")
            logging.info("NN model solving")
            logging.info(
                f"Elapsed Time: {elapsed_time:.6f} seconds"
            )
        else:
            train_loop = (
                range(self.args.num_steps)
                if self.args.verbose
                else tqdm(range(self.args.num_steps))
            )
            losses = self.run_back_prop(train_loop)
        

        solutions_found = self._check_solution()

        self.results[prob_id].update(
            {
                "model_runtime": elapsed_time,
                "model_epochs_ran": self.args.num_steps,
                "solution_throughput": solutions_found/elapsed_time,
            }
        )
        return solutions_found

    def run(self, prob_id: int = 0):
        """Run the experiment."""
        self._initialize_model(prob_id=prob_id)
        solutions_found = self.run_model(prob_id)
        is_verified = solutions_found > 0
        self.results[prob_id].update(
            {
                "num_unique_solutions": solutions_found, 
            }
        )
        
        logging.info("--------------------\n")
        

    def run_all_with_baseline(self):
        """Run all the problems in the dataset given as argument to the Runner."""
        for prob_id in range(len(self.datasets)):
            self.run(prob_id=prob_id)
        if self.args.latency_experiment:
            self.export_results()

    def export_results(self):
        """Export results to a file."""
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{self.problem_type}_{self.dataset_str}_{self.args.num_steps}"
        filename += f"_mse_{self.args.learning_rate}_{self.args.batch_size}.csv"
        filename = os.path.join(self.save_dir, filename)
        df = pd.DataFrame.from_dict(self.results)
        df = df.transpose()
        df.to_csv(filename, sep="\t", index=False)


if __name__ == "__main__":
    runner = Runner(problem_type="sat")
    runner.run_all_with_baseline()
