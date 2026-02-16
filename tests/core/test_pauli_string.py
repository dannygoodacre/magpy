import copy

import pytest
import torch

from magpy import HamOp, PauliString, X, Y, Z, I


@pytest.mark.parametrize('x_mask, z_mask, coeff', [
    (0b1, 0b0, 1.0),
    (0b0, 0b1, 1.0j),
    (0b11, 0b00, 1.0),
    (0b00, 0b11, 2.0),
    (0b10, 0b10, 0.5j),
    (0b101, 0b110, torch.tensor([1.0, 2.0])),
    (1 << 5, 0, 1.0),
])
def test_constructor(x_mask, z_mask, coeff):
    expected_coeff = torch.as_tensor(coeff, dtype=torch.complex128)

    # Act
    result = PauliString(x_mask, z_mask, coeff)

    # Assert
    assert result._x_mask == x_mask
    assert result._z_mask == z_mask

    assert isinstance(result.coeff, torch.Tensor)

    assert result.coeff.allclose(expected_coeff)


@pytest.mark.parametrize('factory, index, expected_x_mask, expected_z_mask', [
    (X, 0, 0b001, 0b000),
    (Y, 0, 0b001, 0b001),
    (Z, 0, 0b000, 0b001),
    (X, 1, 0b010, 0b000),
    (Y, 1, 0b010, 0b010),
    (Z, 1, 0b000, 0b010),
    (X, 2, 0b100, 0b000),
    (Y, 2, 0b100, 0b100),
    (Z, 2, 0b000, 0b100),
])
def test_factories(factory, index, expected_x_mask, expected_z_mask):
    # Act
    p = factory(index)

    # Assert
    assert p._x_mask == expected_x_mask
    assert p._z_mask == expected_z_mask


@pytest.mark.parametrize('p1, p2, expected_result', [
    (X(), X(), PauliString(1, 0, 2)),
    (I(), I(), PauliString(0, 0, 2))
])
def test_add_to_self(p1, p2, expected_result):
    # Act
    result = p1 + p2

    # Assert
    assert result == expected_result


@pytest.mark.parametrize('p1, p2, expected_data', [
    (X(), Y(), {X(): torch.tensor(1, dtype=torch.complex128), Y(): torch.tensor(1, dtype=torch.complex128)}),
    (I(), Z(), {I(): torch.tensor(1, dtype=torch.complex128), Z(): torch.tensor(1, dtype=torch.complex128)}),
    (X(), Y() + Z(), {X(): torch.tensor(1, dtype=torch.complex128), Y(): torch.tensor(1, dtype=torch.complex128), Z(): torch.tensor(1, dtype=torch.complex128)}),
])
def test_add_to_other(p1, p2, expected_data):
    # Act
    result = p1 + p2

    # Assert
    assert isinstance(result, HamOp)

    assert result._data == expected_data


@pytest.mark.parametrize('p, t, kwargs', [
    (X(), None, {}),
    (Y(), torch.tensor(1.0), {'beta': 0.1}),
    (Z(), torch.tensor([1.0, 2.0, 3.0]), {'gamma': 1, 'state': 'test'})
])
def test_call(p, t, kwargs):
    # Act
    result = p(t, **kwargs)

    # Assert
    assert result == p


@pytest.mark.parametrize('p1, p2, expected_equal', [
    (X(), X(), True),
    (X(), Y(), False),
    (PauliString(1, 0, 1.0), PauliString(1, 0, 2.0), True),
    (X(), 'NOT A PAULISTRING', False),
], ids=['identical', 'different_masks', 'different_coeffs', 'wrong_type'])
def test_equality(p1, p2, expected_equal):
    # Act & Assert
    assert (p1 == p2) is expected_equal


def test_hash_consistency():
    # Arrange
    p1 = X()
    p2 = X()
    p3 = Y()

    # Act & Assert
    assert hash(p1) == hash(p2)

    assert hash(p1) != hash(p3)

    d = {p1: 'success'}
    assert d[p2] == 'success'


@pytest.mark.parametrize('p', [
    X(),
    PauliString(1, 1, coeff=torch.tensor([1, 2]))
])
def test_copy(p):
    # Act
    result = copy.copy(p)

    # Assert
    assert result is not p

    assert result == p

    assert result._x_mask == p._x_mask
    assert result._z_mask == p._z_mask


@pytest.mark.parametrize('p, scalar, expected_coeff', [
    (X(), 2, torch.tensor(2, dtype=torch.complex128)),
    (Y(), 0.5j, torch.tensor(0.5j, dtype=torch.complex128)),
    (Z(), torch.tensor([1, 2]), torch.tensor([1, 2], dtype=torch.complex128)),
])
def test_mul_scalar(p, scalar, expected_coeff):
    # Act
    result = p * scalar

    # Assert
    assert isinstance(result, PauliString)

    assert result._x_mask == p._x_mask
    assert result._z_mask == p._z_mask

    assert result.coeff.allclose(torch.as_tensor(expected_coeff, dtype=torch.complex128))


@pytest.mark.parametrize('p1, p2, expected_x_mask, expected_z_mask, expected_coeff', [
    (X(), X(), 0, 0, 1),
    (X(), Y(), 0, 1, 1j),
    (Y(), X(), 0, 1, -1j),
    (Z(), X(), 1, 1, 1j)
], ids=["XX=I", "XY=iZ", "YX=-iZ", "ZX=iY"])
def test_mul_pauli(p1, p2, expected_x_mask, expected_z_mask, expected_coeff):
    # Arrange
    expected_coeff = torch.as_tensor(expected_coeff, dtype=torch.complex128)
    # Act
    result = p1 * p2

    # Assert
    assert isinstance(result, PauliString)

    assert result._x_mask == expected_x_mask
    assert result._z_mask == expected_z_mask

    assert result.coeff.allclose(expected_coeff)


def test_mul_hamop():
    # Arrange
    h = X() + Y()

    p = Z()

    expected_data = {
        Y(): torch.tensor(1j, dtype=torch.complex128),
        X(): torch.tensor(-1j, dtype=torch.complex128)
    }

    # Act
    result = p * h

    # Assert
    assert isinstance(result, HamOp)

    assert result._data == expected_data


def test_mul_callable():
    # Arrange
    def f(t): return torch.sin(t)

    p = X()

    # Act
    result = p * f

    # Assert
    assert isinstance(result, HamOp)


@pytest.mark.parametrize('p', [
    X(),
    PauliString(1, 1, coeff=2),
    PauliString(0, 1, coeff=torch.tensor([1.0, -1.0]))
])
def test_neg(p):
    # Act
    result = -p

    # Assert
    assert isinstance(result, PauliString)

    assert result._x_mask == p._x_mask
    assert result._z_mask == p._z_mask

    assert result.coeff.allclose(-p.coeff)

    assert result is not p


@pytest.mark.parametrize('op, n, expected_x_mask, expected_z_mask, expected_coeff', [
    (X(), 0, 0, 0, 1),
    (Y(), 1, 1, 1, 1),
    (Z(), 2, 0, 0, 1),
    (X(), 3, 1, 0, 1),
    (PauliString(1, 0, 2), 2, 0, 0, 4),
])
def test_pow(op, n, expected_x_mask, expected_z_mask, expected_coeff):
    # Arrange
    expected_coeff = torch.as_tensor(expected_coeff, dtype=torch.complex128)

    # Act
    result = op ** n

    # Assert
    assert isinstance(result, PauliString)

    assert result._x_mask == expected_x_mask
    assert result._z_mask == expected_z_mask

    assert result.coeff.allclose(expected_coeff)


@pytest.mark.parametrize('p, expected_str', [
    (X(), "X0"),
    (Y(), "Y0"),
    (Z(), "Z0"),
    (I(), "I"),
    (PauliString(x_mask=0b01, z_mask=0b11, coeff=1.0), "Y0*Z1"),
    (PauliString(1, 0, coeff=2.0), "2*X0"),
    (PauliString(1, 0, coeff=0.5j), "0.5j*X0"),
])
def test_str_basic(p, expected_str):
    # Act & Assert
    assert str(p) == expected_str


def test_str_with_tensor_coeff():
    # Arrange
    p = PauliString(1, 0, coeff=torch.tensor(1.0, dtype=torch.complex128))
    ps_val = PauliString(1, 0, coeff=torch.tensor(3.14, dtype=torch.complex128))

    # Act & Assert
    assert str(p) == "X0"

    assert "3.14" in str(ps_val)


def test_str_with_identity_toggle(monkeypatch):
    # Arrange
    p = PauliString(x_mask=0b10, z_mask=0b10, coeff=1.0)

    monkeypatch.setattr("magpy.core.pauli_string.get_print_identities", lambda: False)

    # Act & Assert
    assert str(p) == "Y1"

    # # Arrange
    monkeypatch.setattr("magpy.core.pauli_string.get_print_identities", lambda: True)

    # Act & Assert
    assert str(p) == "I0*Y1"


@pytest.mark.parametrize('p1, p2, expected_data', [
    (X(), X(), {}),
    (X(), Z(), {
        X(): torch.tensor(1, dtype=torch.complex128),
        Z(): torch.tensor(-1, dtype=torch.complex128)
    }),
    (PauliString(0b1, 0, 2), X(), {
        X(): torch.tensor(1, dtype=torch.complex128)
    }),
    (PauliString(0b11, 0b00), PauliString(0b00, 0b11), {
        PauliString(0b11, 0b00): torch.tensor(1, dtype=torch.complex128),
        PauliString(0b00, 0b11): torch.tensor(-1, dtype=torch.complex128)
    }),
    (X() * 1j, X() * 0.5j, {
        X(): torch.tensor(0.5j, dtype=torch.complex128)
    })
])
def test_sub(p1, p2, expected_data):
    # Act
    result = p1 - p2

    # Assert
    assert isinstance(result, HamOp)

    assert len(result._data) == len(expected_data)

    for ps, expected_val in expected_data.items():
        assert ps in result._data

        assert result._data[ps].allclose(expected_val)


@pytest.mark.parametrize('p', [
    PauliString(0b1, 0b1, 5),
    PauliString(0b10, 0b00, torch.tensor([1, 2])),
])
def test_as_unit_operator(p):
    # Act
    result = p.as_unit_operator()

    # Assert
    assert isinstance(result, PauliString)

    assert result._x_mask == p._x_mask
    assert result._z_mask == p._z_mask

    assert torch.allclose(result.coeff, torch.tensor(1.0, dtype=torch.complex128))

    assert result is not p


# TODO: Test new expected method compared to old tensor-based version.
