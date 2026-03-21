from magpy import X, Y, Z, frob, timegrid, evolve
import torch

H = X() + Y()
rho0 = X()

i_range = torch.arange(4, 13)

rhoT = torch.zeros((len(i_range), 2, 2), dtype=torch.complex128)

U = torch.matrix_exp(-1j * H.tensor().to(torch.complex128) * 10)
rhoT_ref = U @ rho0.tensor().to(torch.complex128) @ U.conj().transpose(-2, -1)


for index, i in enumerate(i_range):
    tlist = timegrid(0, 10, 0.5**i)

    res, _, _ = evolve(H, rho0, tlist)

    rhoT[index] = res.tensor().to(torch.complex128)

    print(i)

data = {
    "reference": rhoT_ref,
    "result": rhoT,
    "range": i_range
}

torch.save(data, "./data.pt")
