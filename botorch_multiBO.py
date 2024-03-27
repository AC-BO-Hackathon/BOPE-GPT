import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_model

from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective import pareto
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

# Assuming these imports for the missing pieces:
def objective_1(x):
    # Placeholder for the first objective function
    return torch.sin(x).sum(dim=-1)

def objective_2(x):
    # Placeholder for the second objective function
    return torch.cos(x).sum(dim=-1)


# Fixed parts of the setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
MC_SAMPLES = 128
NUM_RESTARTS = 10
RAW_SAMPLES = 512
BATCH_SIZE = 5
dim = 2
bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(device=device, dtype=dtype)
N = 20
iteration_number = 20
refi = torch.tensor([0., 0.], device=device, dtype=dtype)

# Initial Sobol samples
train_x = draw_sobol_samples(bounds=bounds, n=1, q=N).squeeze(0).to(device=device, dtype=dtype)
# Hypothetical objective function evaluations
train_y = torch.stack([objective_1(train_x), objective_2(train_x)], dim=-1).to(device=device, dtype=dtype)

# Initialize models for each objective
models = [SingleTaskGP(train_X=train_x, train_Y=train_y[:, i].unsqueeze(-1)) for i in range(train_y.shape[-1])]

# Fit models
mlls = [ExactMarginalLogLikelihood(model.likelihood, model) for model in models]
for mll in mlls:
    fit_gpytorch_model(mll)

# Combine models into a ModelListGP
model = ModelListGP(*models)

partitioning = FastNondominatedPartitioning(
    ref_point=refi,
    Y=train_y,
    )

# Perform 20 BO iterations
for i in range(iteration_number):
    print("Iteration: "+str(i))
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=refi,
        partitioning=partitioning,
        #sampler=qehvi_sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = candidates.detach()
    
    new_y_obj1 = objective_1(new_x)
    new_y_obj2 = objective_2(new_x)
    new_y = torch.stack([new_y_obj1, new_y_obj2], dim=-1)

    train_x=torch.vstack([train_x,new_x])
    train_y=torch.vstack([train_y,new_y])
    
    models = [SingleTaskGP(train_X=train_x, train_Y=train_y[:, i].unsqueeze(-1)) for i in range(train_y.shape[-1])]

 # Fit models
    mlls = [ExactMarginalLogLikelihood(model.likelihood, model) for model in models]
    for mll in mlls:
        fit_gpytorch_model(mll)
    model = ModelListGP(*models)
        
    partitioning = FastNondominatedPartitioning(
        ref_point=refi,
        Y=train_y,
        )