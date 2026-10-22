import numpy as np
import pytest

from fbmod_extractor.synth import encode_tangent_space
from fbmod_extractor.vertex import (decode_element, format_size, unpack_tangent_space_axis_angle,
                                    unpack_tangent_space_quaternion)

# (packed code, tangent, binormal, normal) as produced by the game's vertex shader
# (CalcTangentSpaceNormal_Quaternion: TBN columns 0/1/2). Generated once from a
# literal transcription of that shader; covers all four slot indices and both
# handedness bits.
KNOWN_ANSWERS = [
    (0x16d92688, [0.918483, -0.025586, 0.394632], [-0.280597, -0.745348, 0.604749], [0.278665, -0.666184, -0.691769]),
    (0xc621fbcd, [0.758489, 0.451638, -0.469806], [0.633085, -0.339642, 0.695591], [-0.15459, 0.825025, 0.54354]),
    (0xa791ffc1, [-0.700361, -0.068838, -0.710462], [-0.072444, -0.983344, 0.166693], [0.710103, -0.168214, -0.683709]),
    (0x705a5661, [0.092441, 0.825226, -0.557186], [-0.287343, -0.513667, -0.808443], [0.953356, -0.234837, -0.189639]),
    (0x6eda1624, [0.490502, 0.817855, 0.300866], [0.251088, 0.197972, -0.947503], [-0.834483, 0.540296, -0.108248]),
    (0xdbcd12c2, [-0.856911, -0.132198, 0.498224], [0.244371, 0.746846, 0.618468], [-0.453857, 0.651724, -0.607675]),
    (0x1600888d, [-0.981906, -0.186347, -0.033693], [0.162595, -0.738419, -0.654447], [-0.097074, 0.648084, -0.755357]),
    (0xb286b60e, [0.705453, -0.700157, -0.110073], [0.670834, 0.709725, -0.215111], [0.228733, 0.07791, 0.970367]),
    (0x339381eb, [0.848343, -0.491619, -0.19653], [-0.512617, -0.85554, -0.072634], [0.132431, -0.162363, 0.977804]),
    (0x181c01b5, [-0.748186, 0.184798, -0.637234], [-0.478274, -0.815884, 0.324942], [0.45986, -0.547889, -0.698818]),
    (0x86c7538c, [0.299413, 0.353242, 0.886325], [0.109049, 0.910185, -0.39959], [-0.947871, 0.216295, 0.234]),
    (0xf9c262ed, [0.746999, -0.077595, 0.660282], [0.569716, 0.586607, -0.575601], [0.122237, -0.753072, -0.646483]),
    (0xbc5a4470, [-0.393606, 0.255243, 0.883134], [0.53805, -0.714974, 0.446446], [0.74537, 0.650894, 0.144085]),
    (0xc2da0d29, [-0.641037, 0.695196, 0.325228], [-0.332104, 0.130777, -0.934133], [0.691938, 0.706824, -0.147044]),
    (0xb7ac9742, [-0.34491, -0.656058, -0.671286], [-0.718345, -0.275836, 0.638667], [-0.604168, 0.702497, -0.376137]),
    (0xc93b82a3, [-0.995341, 0.071802, 0.064343], [0.092972, 0.891531, 0.443316], [0.025533, -0.447233, 0.894053]),
    (0x8362d114, [0.092752, 0.987696, -0.125914], [-0.043144, -0.122354, -0.991548], [-0.994754, 0.097401, 0.031264]),
    (0x20cc0e15, [-0.587146, 0.595835, 0.547941], [-0.684462, -0.726824, 0.056919], [-0.432171, 0.341625, -0.834578]),
    (0xd6f9bd6e, [-0.183831, 0.166624, -0.968733], [-0.917777, -0.381997, 0.108457], [-0.351982, 0.909018, 0.223147]),
    (0x734c7e2f, [0.746774, -0.204179, 0.632961], [-0.645232, -0.453177, 0.615066], [-0.16126, 0.867722, 0.470163]),
]


def test_quaternion_tangent_space_matches_game_shader():
    codes = np.array([k[0] for k in KNOWN_ANSWERS], dtype=np.uint32)
    t, b, n = unpack_tangent_space_quaternion(codes)
    np.testing.assert_allclose(t, [k[1] for k in KNOWN_ANSWERS], atol=2e-5)
    np.testing.assert_allclose(b, [k[2] for k in KNOWN_ANSWERS], atol=2e-5)
    np.testing.assert_allclose(n, [k[3] for k in KNOWN_ANSWERS], atol=2e-5)


def test_quaternion_frames_are_orthonormal():
    codes = np.random.default_rng(3).integers(0, 2**32, 20000, dtype=np.uint64).astype(np.uint32)
    # Keep only valid packings: the three stored components of a unit quaternion have |.|^2 <= 1.
    # (Like the game shader, the decoder clamps invalid ones rather than normalising them.)
    s = np.sqrt(2.0)
    a = (codes >> 22) * (s / 1023) - 1 / s
    b_ = ((codes >> 3) & 0x3FF) * (s / 1023) - 1 / s
    c = ((codes >> 13) & 0x1FF) * (s / 511) - 1 / s
    codes = codes[a * a + b_ * b_ + c * c <= 1.0]
    assert len(codes) > 1000
    t, b, n = unpack_tangent_space_quaternion(codes)
    for v in (t, b, n):
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)
    assert np.abs(np.einsum("ij,ij->i", t, n)).max() < 1e-5
    assert np.abs(np.einsum("ij,ij->i", b, n)).max() < 1e-5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tangent_space_encode_decode_round_trip(seed):
    rng = np.random.default_rng(seed)
    n = rng.normal(size=(300, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    t = np.cross(n, rng.normal(size=(300, 3)))
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    sign = np.where(rng.random(300) < 0.5, -1.0, 1.0)[:, None]
    b = np.cross(n, t) * sign
    t2, b2, n2 = unpack_tangent_space_quaternion(encode_tangent_space(t, b, n))
    assert np.einsum("ij,ij->i", n, n2).min() > 0.999
    assert np.einsum("ij,ij->i", t, t2).min() > 0.999
    assert np.einsum("ij,ij->i", b, b2).min() > 0.999


def test_axis_angle_frames_are_unit_length():
    v = np.random.default_rng(5).random((1000, 4))
    for vec in unpack_tangent_space_axis_angle(v):
        np.testing.assert_allclose(np.linalg.norm(vec, axis=1), 1.0, atol=1e-6)


def test_decode_element_formats():
    half = np.array([[1.5, -2.0, 0.25, 1.0]], dtype="<f2").view(np.uint8).reshape(1, 8)
    np.testing.assert_allclose(decode_element(half, 0x08), [[1.5, -2.0, 0.25, 1.0]])
    ub4n = np.array([[0, 255, 51, 102]], dtype=np.uint8)
    np.testing.assert_allclose(decode_element(ub4n, 0x0D), [[0, 1, 0.2, 0.4]])
    assert decode_element(ub4n, 0x0C).tolist() == [[0, 255, 51, 102]]
    s2n = np.array([[32767, -32767]], dtype="<i2").view(np.uint8).reshape(1, 4)
    np.testing.assert_allclose(decode_element(s2n, 0x13), [[1.0, -1.0]])
    assert format_size(0x08) == 8 and format_size(0x20) == 4 and format_size(0x03) == 12
