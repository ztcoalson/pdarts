import torch

from model_search import Network

torch.set_grad_enabled(False)

clean_model = torch.load("/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/pdarts/results/P-DARTS/clean/search-clean-save-20241222-123359/model-2.pt")
clean_params = clean_model.arch_parameters()

mnist_model = torch.load("/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/pdarts/results/search-mnist1-20250117-085355/model-2.pt")
mnist_params = mnist_model.arch_parameters()

gradpc_model = torch.load("/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/gc_runs/pdarts/arch/gradpc-pdarts-eps=1.0-arch/target_model.pth")
gradpc_params = gradpc_model.arch_parameters()

print(gradpc_params[0])

print(torch.norm(clean_params[0] - mnist_params[0], p=2))
print(torch.norm(clean_params[1] - mnist_params[1], p=2))

print(torch.norm(clean_params[0] - gradpc_params[0], p=2))
print(torch.norm(clean_params[1] - gradpc_params[1], p=2))

