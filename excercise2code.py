import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy import special
import binascii
import scipy.io.wavfile



def bit_array(bits):
    bit_array_result = [0] * bits
    for i in range(bits):
        bit_array_result[i] = random.randint(0, 1)
    return bit_array_result


def bpam_modulation(array_of_bits, A, samples_per_period, bits):
    final_bpam_signal = [0] * bits * samples_per_period
    for i in range(bits):
        if array_of_bits[i] == 1:
            for j in range(samples_per_period):
                final_bpam_signal[i * samples_per_period + j] = A
        else:
            for j in range(samples_per_period):
                final_bpam_signal[i * samples_per_period + j] = -A
    return final_bpam_signal


def bpsk_modulation(array_of_bits, A, fc, samples_per_period, bits, t):
    final_bpsk_signal = [0] * bits * samples_per_period
    for i in range(bits):
        if array_of_bits[i] == 1:
            for j in range(samples_per_period):
                final_bpsk_signal[i * samples_per_period + j] = A * np.cos(
                    2 * np.pi * fc * t[i * samples_per_period + j])
        else:
            for j in range(samples_per_period):
                final_bpsk_signal[i * samples_per_period + j] = -A * np.cos(
                    2 * np.pi * fc * t[i * samples_per_period + j])
    return final_bpsk_signal


def qpsk_modulation(array_of_bits, A, fc, samples_per_period, bits, t):
    final_qpsk_signal = [0] * bits * samples_per_period
    c = [0] * 2
    for i in range(0, bits, 2):
        c1 = array_of_bits[i]
        c2 = array_of_bits[i + 1]
        if c1 == 0 and c2 == 0:
            for j in range(2 * samples_per_period):
                final_qpsk_signal[i * samples_per_period + j] = A * np.cos(
                    2 * np.pi * fc * t[i * samples_per_period + j])
        elif c1 == 0 and c2 == 1:
            for j in range(2 * samples_per_period):
                final_qpsk_signal[i * samples_per_period + j] = A * np.sin(
                    2 * np.pi * fc * t[i * samples_per_period + j])
        elif c1 == 1 and c2 == 0:
            for j in range(2 * samples_per_period):
                final_qpsk_signal[i * samples_per_period + j] = -A * np.sin(
                    2 * np.pi * fc * t[i * samples_per_period + j])
        else:
            for j in range(2 * samples_per_period):
                final_qpsk_signal[i * samples_per_period + j] = -A * np.cos(
                    2 * np.pi * fc * t[i * samples_per_period + j])
    return final_qpsk_signal


def qpsk_modulation1(array_of_bits, A, fc, samples_per_period, bits, t):
    final_qpsk_signal = [0] * bits * samples_per_period
    c = [0] * 2
    for i in range(0, bits, 2):
        c1 = array_of_bits[i]
        c2 = array_of_bits[i + 1]
        if c1 == '0' and c2 == '0':
            for j in range(2 * samples_per_period):
                final_qpsk_signal[i * samples_per_period + j] = A * np.cos(
                    2 * np.pi * fc * t[i * samples_per_period + j])
        elif c1 == '0' and c2 == '1':
            for j in range(2 * samples_per_period):
                final_qpsk_signal[i * samples_per_period + j] = A * np.sin(
                    2 * np.pi * fc * t[i * samples_per_period + j])
        elif c1 == '1' and c2 == '0':
            for j in range(2 * samples_per_period):
                final_qpsk_signal[i * samples_per_period + j] = -A * np.sin(
                    2 * np.pi * fc * t[i * samples_per_period + j])
        else:
            for j in range(2 * samples_per_period):
                final_qpsk_signal[i * samples_per_period + j] = -A * np.cos(
                    2 * np.pi * fc * t[i * samples_per_period + j])
    return final_qpsk_signal


def psk_8_modulation(array_of_bits, A, fc, samples_per_period, bits, t):
    psk_8_final = [0] * bits * samples_per_period
    c = [0] * 3
    for i in range(0, 36, 3):
        c1 = array_of_bits[i]
        c2 = array_of_bits[i + 1]
        c3 = array_of_bits[i + 2]
        if c1 == 0 and c2 == 0 and c3 == 0:
            phase = 0
        elif c1 == 0 and c2 == 0 and c3 == 1:
            phase = np.pi / 4
        elif c1 == 0 and c2 == 1 and c3 == 1:
            phase = np.pi / 2
        elif c1 == 0 and c2 == 1 and c3 == 0:
            phase = 3 * np.pi / 4
        elif c1 == 1 and c2 == 0 and c3 == 0:
            phase = 7 * np.pi / 4
        elif c1 == 1 and c2 == 0 and c3 == 1:
            phase = 6 * np.pi / 4
        elif c1 == 1 and c2 == 1 and c3 == 1:
            phase = 5 * np.pi / 4
        elif c1 == 1 and c2 == 1 and c3 == 0:
            phase = 4 * np.pi / 4
        for j in range(3 * samples_per_period):
            psk_8_final[i * samples_per_period + j] = A * np.cos(2 * np.pi * fc * t[i * samples_per_period + j] + phase)
    return psk_8_final


def qpsk_constellation_diagram(array_of_bits, E, bits):
    constal_qpsk_signal = [0] * bits
    c = [0] * 2
    for i in range(0, bits, 2):
        # c[0] = array_of_bits[i]
        # c[1] = array_of_bits[i + 1]
        # c_gray = create_gray(c, 2)
        c1 = array_of_bits[i]
        c2 = array_of_bits[i + 1]
        if c1 == 0 and c2 == 0:
            for j in range(2):
                constal_qpsk_signal[i + j] = E * np.cos(np.pi / 4) + 1j * E * np.sin(np.pi / 4)
        elif c1 == 0 and c2 == 1:
            for j in range(2):
                constal_qpsk_signal[i + j] = E * np.cos(3 * np.pi / 4) + 1j * E * np.sin(3 * np.pi / 4)
        elif c1 == 1 and c2 == 0:
            for j in range(2):
                constal_qpsk_signal[i + j] = E * np.cos(7 * np.pi / 4) + 1j * E * np.sin(7 * np.pi / 4)
        else:
            for j in range(2):
                constal_qpsk_signal[i + j] = E * np.cos(5 * np.pi / 4) + 1j * E * np.sin(5 * np.pi / 4)
    return constal_qpsk_signal


def qpsk_constellation_diagram1(array_of_bits, E, bits):
    constal_qpsk_signal = [0] * bits
    c = [0] * 2
    for i in range(0, bits, 2):
        # c[0] = array_of_bits[i]
        # c[1] = array_of_bits[i + 1]
        # c_gray = create_gray(c, 2)
        c1 = array_of_bits[i]
        c2 = array_of_bits[i + 1]
        if c1 == '0' and c2 == '0':
            for j in range(2):
                constal_qpsk_signal[i + j] = E * np.cos(np.pi / 4) + 1j * E * np.sin(np.pi / 4)
        elif c1 == '0' and c2 == '1':
            for j in range(2):
                constal_qpsk_signal[i + j] = E * np.cos(3 * np.pi / 4) + 1j * E * np.sin(3 * np.pi / 4)
        elif c1 == '1' and c2 == '0':
            for j in range(2):
                constal_qpsk_signal[i + j] = E * np.cos(7 * np.pi / 4) + 1j * E * np.sin(7 * np.pi / 4)
        else:
            for j in range(2):
                constal_qpsk_signal[i + j] = E * np.cos(5 * np.pi / 4) + 1j * E * np.sin(5 * np.pi / 4)
    return constal_qpsk_signal


def create_AWGN(SNR, A, Tb, bits):
    n01 = (A ** 2) * Tb / 10 ** (SNR / 10)
    X1 = [0] * bits
    Y1 = [0] * bits
    noise1 = [0] * bits
    for i in range(bits):
        X1[i] = np.random.normal(0, math.sqrt(n01 / 2))
        Y1[i] = np.random.normal(0, math.sqrt(n01 / 2))
        noise1[i] = complex(X1[i], Y1[i])
    return noise1


def decode(bit_array):
    return ''.join(chr(int(bit_array[i * 8:i * 8 + 8], 2)) for i in range(len(bit_array) // 8))


def arr_conv_wav(symbol_stream, bit_stream, col):
    for i in range(0, bit_stream.size, col):
        if (bit_stream[i] == 0) and (bit_stream[i + 1] == 0):
            symbol_stream[int(i / 2)] = 0
        elif (bit_stream[i] == 0) and (bit_stream[i + 1] == 1):
            symbol_stream[int(i / 2)] = 1
        elif (bit_stream[i] == 1) and (bit_stream[i + 1] == 1):
            symbol_stream[int(i / 2)] = 2
        else:
            symbol_stream[int(i / 2)] = 3

    return symbol_stream

# Ακαρέπης Ανδρέας
# Α.Μ:03117058
Tb = 0.2
Tall = Tb * 36
A = 9  # (A.M.=03117180, αρα Α=1+8+0 = 9)

# erwthma 1a
t = np.linspace(0, Tall, 36 * 500)
sig_transmitted = [0] * np.size(t)
ran_array = [0] * 36
# επιλέγεται τυχαία το κάθε bit της ακολουθίας και διαμορφώνεται το BPAM διαμορφωμένο σήμα
for i in range(36):
    c = random.randint(0, 1)
    for j in range(500):
        if c == 1:
            sig_transmitted[i * 500 + j] = A
            ran_array[i] = A
        else:
            sig_transmitted[i * 500 + j] = (-1) * A
            ran_array[i] = -A
plt.plot(t, sig_transmitted)
plt.grid()
plt.xlabel("Time(s)")
plt.ylabel("Voltage(V)")
plt.title("Binary PAM of random bit stream")
plt.show()
# erwthma 1b
plt.plot([-A * math.sqrt(Tb), A * math.sqrt(Tb)], [0, 0], 'ro')
plt.axis([-10, 10, -0.1, 0.5])
plt.grid()
plt.title("Constellation diagram of B-PAM")
plt.show()
# erwthma 1c
SNR1 = 6
SNR2 = 12
n01 = (A ** 2) * Tb / 10 ** (SNR1 / 10)  # υπολογίζονται οι αποκλίσεις των δύο θορύβων με βάση τον θεωρητικό τύπο
n02 = (A ** 2) * Tb / 10 ** (SNR2 / 10)
X1 = [0] * 36
Y1 = [0] * 36
noise1 = [0] * 36
sig_tr_with_noise1 = [0] * np.size(t)
X2 = [0] * 36
Y2 = [0] * 36
noise2 = [0] * 36
sig_tr_with_noise2 = [0] * np.size(t)
for i in range(36):
    X1[i] = np.random.normal(0, math.sqrt(n01 / 2))
    Y1[i] = np.random.normal(0, math.sqrt(n01 / 2))
    noise1[i] = complex(X1[i], Y1[i])
for i in range(36):
    for j in range(500):
        sig_tr_with_noise1[i * 500 + j] = sig_transmitted[i * 500 + j] + X1[i]  # σήμα με θόρυβο με snr=6dB
for i in range(36):
    X2[i] = np.random.normal(0, math.sqrt(n02 / 2))
    Y2[i] = np.random.normal(0, math.sqrt(n02 / 2))
    noise2[i] = complex(X2[i], Y2[i])
for i in range(36):
    for j in range(500):
        sig_tr_with_noise2[i * 500 + j] = sig_transmitted[i * 500 + j] + X2[i]  # σήμα με θόρυβο με snr=12dB
plt.grid()
plt.xlabel("Time(s)")
plt.ylabel("Voltage(V)")
plt.title("Adding AWGN with SNR = 6 to BPAM signal ")
plt.plot(t, sig_tr_with_noise1)
plt.show()
plt.grid()
plt.xlabel("Time(s)")
plt.ylabel("Voltage(V)")
plt.title("Adding AWGN with SNR = 12 to BPAM signal ")
plt.plot(t, sig_tr_with_noise2)
plt.show()
# erwthma 1d
# κάνουμε τα διαγράμματα αστερισμού των δύο προηγούμενων σημάτων
real1 = [0] * 36
real2 = [0] * 36
for i in range(36):
    real1[i] = ran_array[i] * math.sqrt(Tb) + X1[i]  # Στην ενέργεια του κάθε σημείου προσθέτουμε θόρυβο και
    real2[i] = ran_array[i] * math.sqrt(Tb) + X2[i]  # φτιάχνουμε το διάγραμμα αστερισμού
plt.grid(True)
plt.title("Constellation diagram, with SNR = 6")
plt.scatter(real1, Y1, c='b')
plt.axis([-10, 10, -20, 20])
plt.show()
plt.grid(True)
plt.title("Constellation diagram, with SNR = 6")
plt.scatter(real2, Y2, color='g')
plt.axis([-10, 10, -20, 20])
plt.show()
# erwthhma 1e
Yax = [0] * 16
Xax = [0] * 16
SNR = np.linspace(0, 16, 16 * 5000)
Ber_th = [0] * 80000
for i in range(80000):
    Ber_th[i] = 0.5 * special.erfc(math.sqrt(10 ** (i * 0.0002 / 10)))
for snr in range(16):
    Xax[snr] = snr
    n0 = (A ** 2) * Tb / 10 ** (snr / 10)
    correct = 0
    for i in range(100000):
        bit = random.randint(0, 1)
        if bit == 1:
            bit_tr = A
        else:
            bit_tr = -A
        x1 = np.random.normal(0, math.sqrt(n0 / 2))
        x2 = np.random.normal(0, math.sqrt(n0 / 2))
        bit_noise = bit_tr * math.sqrt(Tb) + x1
        if bit_tr * bit_noise >= 0:  # αν το αρχικό σήμα και το σήμα μετά τον θόρυβο έχουν ενέργειες με ίδιο πρόσημο
            correct += 1  # τότε ο θόρυβος δεν επηρεάζει την αποδιαμόρφωση
    Yax[snr] = (100000 - correct) / 100000
plt.yscale('log')
plt.margins(x=0.000)
plt.margins(y=0.015)
plt.title("Bit Error Rate diagram")
plt.xlabel("SNR")
plt.scatter(Xax, Yax, c='r')
plt.plot(SNR, Ber_th)
plt.show()
# ερωτημα 2
if A % 2 == 0:
    fc = 2
else:
    fc = 3
# παραγωγή ακολουθίας με άσσους και μηδενικά
array_of_bits1 = [0] * 36
array_of_bits1 = bit_array(36)
t1 = np.linspace(0, Tall, 36 * 500)
bpam_array = bpam_modulation(array_of_bits1, 1, 500, 36)
plt.grid()
plt.title("Bitstream")
plt.plot(t1, bpam_array)
plt.show()
# BPSK modulation
plt.grid()
plt.title("BPSK modulated bitstream")
plt.xlabel("t(sec)")
plt.ylabel("Voltage(V")
bpsk_signal = bpsk_modulation(array_of_bits1, A, fc, 500, 36, t1)
plt.plot(t1, bpsk_signal)
plt.show()
# QPSK modulation
qpsk_signal = qpsk_modulation(array_of_bits1, A, fc, 500, 36, t1)
plt.grid()
plt.title("QPSK modulated bitstream")
plt.xlabel("t(sec)")
plt.ylabel("Voltage(V")
plt.plot(t1, qpsk_signal)
plt.show()
# 8-PSK modulation
psk_8_signal = psk_8_modulation(array_of_bits1, A, fc, 500, 36, t1)
plt.grid()
plt.title("8-PSK modulated bitstream")
plt.xlabel("t(sec)")
plt.ylabel("Voltage(V")
plt.plot(t1, psk_8_signal)
plt.show()

# ερωτημα 3
E = A * math.sqrt(Tb)
array_of_bits1 = [0] * 36
array_of_bits1 = bit_array(36)
qpsk_const_diagram_signal = qpsk_constellation_diagram(array_of_bits1, E, 36)
# a
# Το διάγραμμα αστερισμού ενός QPSK modulated signal αποτελείται από 4 σημεία
x = np.array([E * np.cos(np.pi / 4), E * np.cos(3 * np.pi / 4), E * np.cos(5 * np.pi / 4), E * np.cos(7 * np.pi / 4)])
y = np.array([E * np.sin(np.pi / 4), E * np.sin(3 * np.pi / 4), E * np.sin(5 * np.pi / 4), E * np.sin(7 * np.pi / 4)])
plt.grid()
plt.title("Constellation diagram of QPSK modulated signal")
for i in range(36):
    plt.scatter(qpsk_const_diagram_signal[i].real, qpsk_const_diagram_signal[i].imag, c='b')
plt.show()
# b
# παραγωγή και προσθήκη στο qpsk σήμα και αναπαράσταση διαγράμματος αστερισμού
SNR1 = 6
SNR2 = 12
noise1 = create_AWGN(SNR1, A, Tb, 36)
noise2 = create_AWGN(SNR2, A, Tb, 36)
plt.title("Constellation diagram of QPSK modulated signal with AWGN, with SNR=6")
for i in range(36):
    plt.scatter(qpsk_const_diagram_signal[i].real + noise1[i].real, qpsk_const_diagram_signal[i].imag + noise1[i].imag,
                c='g')
plt.grid(True)
plt.axis([-3, 3, -3, 3])
plt.show()
for i in range(36):
    plt.scatter(qpsk_const_diagram_signal[i].real + noise2[i].real, qpsk_const_diagram_signal[i].imag + noise2[i].imag,
                c='r')
plt.grid()
plt.title("Constellation diagram of QPSK modulated signal with AWGN, with SNR=12")
plt.axis([-3, 3, -3, 3])
plt.show()
# c
Yax = [0] * 16
Xax = [0] * 16
SNR = np.linspace(0, 15, 15 * 5000)
SNR_PR = [0] * 16
Ber_th = [0] * 75000
for i in range(75000):
    Ber_th[i] = 0.5 * special.erfc(math.sqrt(10 ** (i * 0.0002 / 10)))
X_signal = [0] * 10 ** 5
Y_signal = [0] * 10 ** 5
qpsk_diagram_signal = [0] * 10 ** 5
for i in range(10 ** 5):
    X_signal[i] = np.round(np.random.rand(1)) - 0.5
    Y_signal[i] = np.round(np.random.rand(1)) - 0.5
    qpsk_diagram_signal[i] = math.sqrt(E) * 2 * (X_signal[i] + 1j * Y_signal[i])
ber_x = [0] * 16
ber_y = [0] * 16
for snr in range(16):
    SNR_PR[snr] = snr
    X = math.sqrt(E) * np.random.normal(0, 1, size=10 ** 5)
    Y = math.sqrt(E) * np.random.normal(0, 1, size=10 ** 5)
    noise = 1 / (math.sqrt(2 * 10 ** (snr / 10))) * (X + 1j * Y)
    correct_x = 0
    correct_y = 0
    for i in range(10 ** 5):
        r_real = np.sign(qpsk_diagram_signal[i].real + noise[i].real)
        r_imag = np.sign(qpsk_diagram_signal[i].imag + noise[i].imag)
        if r_real == np.sign(qpsk_diagram_signal[i].real):
            correct_x += 1  # αν το προσήμο των ενεργειών πριν και μετά τον θόρυβο
        if r_imag == np.sign(qpsk_diagram_signal[i].imag):  # είναι ίσο, τότε ο θόρυβος δεν επηρεάζει το σήμα
            correct_y += 1
    ber_x[snr] = 0.5 * ((10 ** 5 - correct_x) / (10 ** 5) + (10 ** 5 - correct_y) / (10 ** 5))
plt.grid()
plt.title("Bit Error Rate diagram")
plt.xlabel("SNR")
plt.yscale('log')
plt.scatter(SNR_PR, ber_x, c='r')
plt.plot(SNR, Ber_th)
plt.show()
#d
filename = 'clarke_relays_{}.txt'.format('even' if A % 2 == 0 else 'odd')
f = open(filename, "rb")
text = f.read()
f.seek(0, 0)
size = len(text)
ascii_array = []
for i in range(size):
    word = f.read(1)
    string_bin_of_word = bin(int(binascii.hexlify(word), 16))
    ascii_array.append(int(string_bin_of_word, 2))   # αντιστοιχεία κειμένου και ascii κωδικοποίησης
f.close()
# mid-rise quantization
bits = 8
levels = 2 ** bits - 1
delta = (max(ascii_array)) / levels
h = [0] * size
data_ascii_quantized = [0] * size
for i in range(size):
    h[i] = 2 * (ascii_array[i] // delta) + 1
    data_ascii_quantized[i] = (delta / 2) * h[i]
axis_x = np.arange(0, size, 1)
plt.grid()
plt.title("Quantized signal of ascii_array expressed by bit_array")
plt.stem(axis_x, data_ascii_quantized)
plt.show()
# qpsk

binarray = ['{:08b}'.format(x) for x in ascii_array]
quantized_signal_bits = ''.join(binarray)  # αναπαράσταση του κειμένου στο δυαδικό σύστημα
size = size * 8
axis_x2 = np.arange(0, size, 1)
qpsk_of_quantized_signal = qpsk_modulation1(quantized_signal_bits, 1, fc, 1, size, axis_x2)
plt.grid()
plt.title("QPSK of quantized signal")
plt.plot(axis_x2 / 300, qpsk_of_quantized_signal)
plt.show()
# qpsk with noise
snr1 = 6
snr2 = 12
const_qpsk = qpsk_constellation_diagram1(quantized_signal_bits, math.sqrt(Tb), size)
noise1 = create_AWGN(snr1, 1, Tb / 2, size)  # προσθήκη θορύβου
noise2 = create_AWGN(snr2, 1, Tb / 2, size)
const_qpsk1 = [0] * size
const_qpsk2 = [0] * size
for i in range(size):
    const_qpsk1[i] = const_qpsk[i].real + noise1[i].real + 1j * const_qpsk[i].imag + noise1[i].imag
    plt.scatter(const_qpsk[i].real + noise1[i].real, const_qpsk[i].imag + noise1[i].imag, c='b')
plt.grid()
plt.title("constellation diagram of QSP signal after addin AWGN with SNR=6dB")
plt.show()
for i in range(size):
    const_qpsk2[i] = const_qpsk[i].real + noise2[i].real + 1j * const_qpsk[i].imag + noise2[i].imag
    plt.scatter(const_qpsk[i].real + noise2[i].real, const_qpsk[i].imag + noise2[i].imag, c='g')
plt.grid()
plt.title("constellation diagram of QSP signal after addin AWGN with SNR=12dB")
plt.show()
# σύγκριση θεωρητικών και πειραματικών τιμών του BER
ber_th1 = 0.5 * special.erfc(math.sqrt(10 ** (snr1 / 10)))
ber_th2 = 0.5 * special.erfc(math.sqrt(10 ** (snr2 / 10)))
s_real = math.sqrt(math.sqrt(Tb)) * 2 * np.round(np.random.rand(1)) - 0.5
s_imag = math.sqrt(math.sqrt(Tb)) * 2 * np.round(np.random.rand(1)) - 0.5
X1 = 1 / (math.sqrt(2 * 10 ** (snr1 / 10))) * math.sqrt(math.sqrt(Tb)) * np.random.normal(0, 1, size)
Y1 = 1 / (math.sqrt(2 * 10 ** (snr1 / 10))) * math.sqrt(math.sqrt(Tb)) * np.random.normal(0, 1, size)
X2 = 1 / (math.sqrt(2 * 10 ** (snr2 / 10))) * math.sqrt(math.sqrt(Tb)) * np.random.normal(0, 1, size)
Y2 = 1 / (math.sqrt(2 * 10 ** (snr2 / 10))) * math.sqrt(math.sqrt(Tb)) * np.random.normal(0, 1, size)
correct_x1 = 0
correct_y1 = 0
correct_x2 = 0
correct_y2 = 0
for i in range(size):
    s_real = const_qpsk[i].real
    s_imag = const_qpsk[i].imag
    r_real1 = np.sign(s_real + X1[i])
    r_imag1 = np.sign(s_imag + Y1[i])
    if r_real1 == np.sign(s_real):
        correct_x1 += 1
    if r_imag1 == np.sign(s_imag):
        correct_y1 += 1
    r_real2 = np.sign(s_real + X2[i])
    r_imag2 = np.sign(s_imag + Y2[i])
    if r_real2 == np.sign(s_real):
        correct_x2 += 1
    if r_imag2 == np.sign(s_imag):
        correct_y2 += 1
ber_pr1 = 0.5 * (size - correct_x1) / size + 0.5 * (size - correct_y1) / size
ber_pr2 = 0.5 * (size - correct_x2) / size + 0.5 * (size - correct_y2) / size
print("ber_theoritical_for_snr_6", ber_th1)
print("ber_practical_for_snr_6", ber_pr1)
print("ber_theoritical_for_snr_12", ber_th2)
print("ber_practical_for_snr_12", ber_pr2)
# αποδιαμόρφωση και ανακατασκευή κειμένου
bitstream1 = []
bitstream2 = []
for i in range(0, size, 2):   # ελέγχουμε σε ποιό τεταρτημόριο βρισκόμαστε και προσθέτουμε τα αντίστοιχα bits
    if const_qpsk1[i].imag < 0:   # ανάλογα με το mapping με Gray code του QPSK
        bitstream1.append('1')
    else:
        bitstream1.append('0')
    if const_qpsk1[i].real < 0:
        bitstream1.append('1')
    else:
        bitstream1.append('0')
    if const_qpsk2[i].imag < 0:
        bitstream2.append('1')
    else:
        bitstream2.append('0')
    if const_qpsk2[i].real < 0:
        bitstream2.append('1')
    else:
        bitstream2.append('0')
bitstream1 = ''.join(bitstream1)
bitstream2 = ''.join(bitstream2)
print(bitstream1)
print(bitstream2)
result1 = open('text_with_snr_6{}.txt'.format('even' if A % 2 == 0 else 'odd'), 'w')
text1 = decode(bitstream1)
print(text1)
result1.write(text1)
result1.close()
result2 = open('text_with_snr_12{}.txt'.format('even' if A % 2 == 0 else 'odd'), 'w')
text2 = decode(bitstream2)
result2.write(text2)
print(text2)
result2.close()

#ερωτημα 4
if A % 2 == 0:
    samplerate, data = scipy.io.wavfile.read("soundfile2_lab2.wav", mmap=False)
else:
    samplerate, data = scipy.io.wavfile.read("soundfile1_lab2.wav", mmap=False)


bits = 8  # Number of bits according to fm
levels = 2 ** bits-1  # Number of levels for the quantizer
data_max = np.amax(data)
delta = 2 * data_max / levels

fig = plt.figure(figsize=(10, 8))
x = np.arange(0, data.size)
plt.plot(x, data, '.')
plt.title("acoustic_signal")
plt.grid()
plt.show()
# mid riser quantization
size = data.size
bits = 8
levels = 2 ** bits - 1
delta = (max(data)) / levels
data_acoustic_quantized = delta * (np.floor(data // delta) + 1)
# h = [0] * size
# data_acoustic_quantized = [0] * size
# for i in range(size):
#    h[i] = 2 * np.floor(data[i] // delta) + 1
#    data_acoustic_quantized[i] = ((delta / 2) * h[i]).astype('int16')
size = len(data_acoustic_quantized)
x = np.arange(0, size, 1)
print(size)
plt.grid()
plt.title("quantized_acoustic_signal")
# plt.axis([0, 1000, 0, 250])
plt.plot(x, data_acoustic_quantized)
plt.show()


# We create the mid-riser quantizer according to its math function
q_o = ((np.floor(data / delta) + 0.5) * delta).astype('uint8')
q_o = np.minimum(q_o, (data_max - delta / 2).astype('uint8'))  # Takes care of the max value

data_bit_stream = np.unpackbits(q_o)

data_symbol_stream = np.empty(int(data_bit_stream.size / 2), dtype=int)
data_symbol_stream = arr_conv_wav(data_symbol_stream, data_bit_stream, 2)

# QPSK modulation
data_qpsk = np.empty(0, dtype=complex)
phase = (data_symbol_stream * 2 + 1) * np.pi / 4
data_qpsk = A * (np.cos(phase) + 1j * (np.sin(phase)))

# Add noise
noise = create_AWGN(4, A, Tb, len(data_qpsk))
r_qpsk_4dB = data_qpsk + noise

noise = create_AWGN(14, A, Tb, len(data_qpsk))
r_qpsk_14dB = data_qpsk + noise

# Plot Constellation
fig = plt.figure(figsize=(8, 6))

plt.plot(r_qpsk_4dB.real, r_qpsk_4dB.imag, 'b.')
plt.title('constellation diagram of quantized_acoustic_signal with AWGN with SNR = 4dB')
plt.grid()
plt.show()
plt.plot(r_qpsk_14dB.real, r_qpsk_14dB.imag, 'g.')
plt.title('constellation diagram of quantized_acoustic_signal with AWGN with SNR = 14dB')
plt.grid()
plt.show()

# demodulation for 4db
r_symbol_wav_4db = np.empty(data_qpsk.size, dtype=int)
mask = ((r_qpsk_4dB.real > 0) & (r_qpsk_4dB.imag > 0))
r_symbol_wav_4db[mask] = 0
mask = ((r_qpsk_4dB.real < 0) & (r_qpsk_4dB.imag > 0))
r_symbol_wav_4db[mask] = 1
mask = ((r_qpsk_4dB.real < 0) & (r_qpsk_4dB.imag < 0))
r_symbol_wav_4db[mask] = 2
mask = ((r_qpsk_4dB.real > 0) & (r_qpsk_4dB.imag < 0))
r_symbol_wav_4db[mask] = 3

# demodulation for 14db
r_symbol_wav_14db = np.empty(data_qpsk.size, dtype=int)
mask = ((r_qpsk_14dB.real > 0) & (r_qpsk_14dB.imag > 0))
r_symbol_wav_14db[mask] = 0
mask = ((r_qpsk_14dB.real < 0) & (r_qpsk_14dB.imag > 0))
r_symbol_wav_14db[mask] = 1
mask = ((r_qpsk_14dB.real < 0) & (r_qpsk_14dB.imag < 0))
r_symbol_wav_14db[mask] = 2
mask = ((r_qpsk_14dB.real > 0) & (r_qpsk_14dB.imag < 0))
r_symbol_wav_14db[mask] = 3

r_bit_wav_14db = [0] * (r_symbol_wav_14db.size)

for i in range(0, r_symbol_wav_14db.size,2):
    if r_symbol_wav_14db[i] == 0:
        r_bit_wav_14db[ i] = 0
        r_bit_wav_14db[  i + 1] = 0
    elif r_symbol_wav_14db[i] == 1:
        r_bit_wav_14db[ i] = 0
        r_bit_wav_14db[ i + 1] = 1
    elif r_symbol_wav_14db[i] == 2:
        r_bit_wav_14db[ i] = 1
        r_bit_wav_14db[ i + 1] = 1
    else:
        r_bit_wav_14db[ i] = 1
        r_bit_wav_14db[ i + 1] = 0

r_bit_wav_4db = np.empty(2 * r_symbol_wav_4db.size, dtype=int)
for i in range(0, r_symbol_wav_4db.size):
    if r_symbol_wav_4db[i] == 0:
        r_bit_wav_4db[2 * i] = 0
        r_bit_wav_4db[2 * i + 1] = 0
    elif r_symbol_wav_4db[i] == 1:
        r_bit_wav_4db[2 * i] = 0
        r_bit_wav_4db[2 * i + 1] = 1
    elif r_symbol_wav_14db[i] == 2:
        r_bit_wav_4db[2 * i] = 1
        r_bit_wav_4db[2 * i + 1] = 1
    else:
        r_bit_wav_4db[2 * i] = 1
        r_bit_wav_4db[2 * i + 1] = 0

music_bytes = np.packbits(r_bit_wav_14db)
if A % 2 == 0:
    scipy.io.wavfile.write("soundfile2_recreated.wav", samplerate, music_bytes)
else:
    scipy.io.wavfile.write('soundfile1_recreated.wav', samplerate, music_bytes)

