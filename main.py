"""
================================================================================
ПОЛНАЯ МОДЕЛЬ СИСТЕМЫ СВЯЗИ BPSK В УСЛОВИЯХ СЛОЖНОЙ ПОМЕХОВОЙ ОБСТАНОВКИ
================================================================================
Тема: "Организация каналов связи в условиях сложной помеховой обстановки.
       Методы борьбы с преднамеренными помехами."

Моделируемые помехи:
  1. Прицельная тональная (Tone Jammer)
  2. Заградительная шумовая (Barrage Jammer)
  3. Имитационная (Spoofing)

Методы борьбы:
  - Адаптивная частотная режекция (Notch Filter)
  - Прямое расширение спектра (DSSS, код Баркера-11)
  - Псевдослучайная перестройка частоты (FHSS)
  - Аутентификация на физическом уровне (Watermarking)

Оценка качества: Bit Error Rate (BER)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ================= 1. ИНИЦИАЛИЗАЦИЯ ПАРАМЕТРОВ =================
np.random.seed(42)

K = 20                      # Число бит
Fs = 1e6                    # Частота дискретизации, Гц
T_s = 5e-5                  # Длительность символа, с
f0 = 50e3                   # Несущая частота, Гц
E_b = 1.0                   # Энергия бита
A = np.sqrt(2 * E_b / T_s)  # Амплитуда сигнала

N_sym = int(T_s * Fs)       # Отсчётов на символ (50)
N = K * N_sym               # Общее число отсчётов (1000)

t = np.arange(N) / Fs       # Временная сетка

print("=" * 60)
print("МОДЕЛИРОВАНИЕ СИСТЕМЫ СВЯЗИ BPSK")
print("=" * 60)
print(f"Число бит K = {K}")
print(f"Длительность символа T_s = {T_s*1e6:.0f} мкс")
print(f"Несущая частота f_0 = {f0*1e-3:.0f} кГц")
print(f"Амплитуда сигнала A = {A:.1f}")

# ================= 2. ГЕНЕРАЦИЯ ПОЛЕЗНОГО BPSK-СИГНАЛА =================
d_k = np.random.randint(0, 2, K)
s_k = 2 * d_k - 1
modulated = np.repeat(s_k, N_sym)

carrier = np.cos(2 * np.pi * f0 * t)
x_t = A * modulated * carrier

print(f"Переданные биты: {d_k}")

# ================= 3. МОДЕЛИРОВАНИЕ ПОМЕХ =================

# 3.1 Тональная помеха
A_J = A * 4.0                # Амплитуда в 4 раза больше сигнала → JSR ≈ +12 дБ
theta = np.random.uniform(0, 2 * np.pi)
f_J = f0
J_T = A_J * np.cos(2 * np.pi * f_J * t + theta)

# 3.2 Шумовая помеха
sigma_J_sq = A**2 * 3.0      # Мощность шума в 3 раза больше мощности сигнала
J_N = np.sqrt(sigma_J_sq) * np.random.randn(N)

# 3.3 Тепловой шум — очень маленький, чтобы не маскировать эффект помех
sigma_n_sq = A**2 * 0.002    # Уменьшен в 10 раз
n_t = np.sqrt(sigma_n_sq) * np.random.randn(N)

# 3.4 Имитационная помеха (спуфинг)
A_S = A * 0.9
tau_S_samples = 5
phi_S = np.pi / 6

d_k_spoof = 1 - d_k
s_k_spoof = 2 * d_k_spoof - 1
modulated_spoof = np.repeat(s_k_spoof, N_sym)
carrier_spoof = np.cos(2 * np.pi * f0 * t + phi_S)
J_S = np.zeros(N)
if tau_S_samples >= 0:
    J_S[tau_S_samples:] = A_S * modulated_spoof[:N-tau_S_samples] * carrier_spoof[tau_S_samples:]

# 3.5 Принятая смесь
y_t = x_t + J_T + J_N + J_S + n_t

# Расчёт JSR
P_signal = A**2 / 2
P_tone = A_J**2 / 2
P_noise = sigma_J_sq
P_spoof = A_S**2 / 2
P_thermal = sigma_n_sq

JSR_tone_dB = 10 * np.log10(P_tone / P_signal)
JSR_noise_dB = 10 * np.log10(P_noise / (P_signal + P_thermal))
JSR_spoof_dB = 10 * np.log10(P_spoof / P_signal)

print(f"JSR тональной = {JSR_tone_dB:.1f} дБ")
print(f"JSR шумовой  = {JSR_noise_dB:.1f} дБ")
print(f"JSR спуфинга = {JSR_spoof_dB:.1f} дБ")

# ================= 4. МЕТОД 1: АДАПТИВНАЯ ЧАСТОТНАЯ РЕЖЕКЦИЯ =================
Y_f = np.fft.fft(y_t)
freqs = np.fft.fftfreq(N, 1/Fs)
spectrum_mag = np.abs(Y_f)

mean_spec = np.mean(spectrum_mag)
std_spec = np.std(spectrum_mag)
beta = 2.5
lambda_th = mean_spec + beta * std_spec

jammed_indices = np.where(spectrum_mag > lambda_th)[0]
mask = np.ones(N, dtype=complex)
mask[jammed_indices] = 0

Y_f_filtered = Y_f * mask
y_hat = np.fft.ifft(Y_f_filtered).real

print(f"Режекция: порог λ={lambda_th:.1f}, подавлено {len(jammed_indices)} из {N} компонент")

# ================= 5. ДЕМОДУЛЯЦИЯ И BER ДЛЯ БАЗОВЫХ СЦЕНАРИЕВ =================
def demodulate_BPSK(signal_in, t_in, f0_in, K_in, N_sym_in):
    decoded = np.zeros(K_in)
    for k in range(K_in):
        start = k * N_sym_in
        end = (k + 1) * N_sym_in
        M_k = np.sum(signal_in[start:end] * np.cos(2 * np.pi * f0_in * t_in[start:end]))
        decoded[k] = 1 if M_k > 0 else 0
    return decoded

bits_clean = demodulate_BPSK(x_t, t, f0, K, N_sym)
bits_jammed = demodulate_BPSK(y_t, t, f0, K, N_sym)
bits_filtered = demodulate_BPSK(y_hat, t, f0, K, N_sym)

def count_errors(orig, dec):
    return np.sum(orig != dec)

err_clean = count_errors(d_k, bits_clean)
err_jammed = count_errors(d_k, bits_jammed)
err_filtered = count_errors(d_k, bits_filtered)

ber_clean = err_clean / K
ber_jammed = err_jammed / K
ber_filtered = err_filtered / K

# Правильный расчёт E_b/N_0 с учётом новых параметров
SNR_linear = (A**2 * T_s / 2) / (sigma_n_sq / Fs)
ber_theory = 0.5 * erfc(np.sqrt(SNR_linear))
print(f"E_b/N_0 = {SNR_linear:.1f} ({10*np.log10(SNR_linear):.1f} дБ)")
print(f"Теория BPSK (АБГШ):  BER = {ber_theory:.2e}")

print(f"\n{'='*60}")
print(f"БАЗОВЫЕ РЕЗУЛЬТАТЫ")
print(f"{'='*60}")
print(f"Идеальный канал:     ошибок {err_clean}/{K}, BER = {ber_clean:.3f}")
print(f"С помехами, без защ.: ошибок {err_jammed}/{K}, BER = {ber_jammed:.3f}")
print(f"После режекции:      ошибок {err_filtered}/{K}, BER = {ber_filtered:.3f}")
print(f"Теория BPSK (АБГШ):  BER = {ber_theory:.4f}")

# ================= 6. МЕТОД 2: DSSS (КОД БАРКЕРА-11) =================
L_c = 11
samples_per_chip = N_sym // L_c          # 50 // 11 = 4
effective_chips = L_c * samples_per_chip # 44
N_dsss_total = effective_chips * K       # 880

t_dsss = t[:N_dsss_total]
carrier_dsss = carrier[:N_dsss_total]
modulated_dsss = modulated[:N_dsss_total]

barker_11 = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1])
chip_sequence = np.repeat(barker_11, samples_per_chip)
spreading_seq = np.tile(chip_sequence, K)

x_dsss = A * modulated_dsss * spreading_seq * carrier_dsss
y_dsss = x_dsss + J_T[:N_dsss_total] + J_N[:N_dsss_total] + J_S[:N_dsss_total] + n_t[:N_dsss_total]
y_despread = y_dsss * spreading_seq

decoded_dsss = np.zeros(K)
for k in range(K):
    start = k * effective_chips
    end = (k + 1) * effective_chips
    M_k = np.sum(y_despread[start:end] * np.cos(2 * np.pi * f0 * t_dsss[start:end]))
    decoded_dsss[k] = 1 if M_k > 0 else 0

err_dsss = count_errors(d_k, decoded_dsss)
ber_dsss = err_dsss / K
Gp_dB = 10 * np.log10(L_c)

# ================= 7. МЕТОД 3: FHSS (МЕДЛЕННАЯ ППРЧ) =================
N_ch = 10
N_jam = 2
delta_f = 5e3
f_min = f0 - (N_ch // 2) * delta_f

np.random.seed(123)
hop_seq = np.random.randint(0, N_ch, K)
jammed_channels = np.random.choice(N_ch, N_jam, replace=False)

x_fhss = np.zeros(N)
for k in range(K):
    start = k * N_sym
    end = (k + 1) * N_sym
    f_hop = f_min + hop_seq[k] * delta_f
    x_fhss[start:end] = A * s_k[k] * np.cos(2 * np.pi * f_hop * t[start:end])

J_T_fhss = np.zeros(N)
for ch in jammed_channels:
    f_jam_ch = f_min + ch * delta_f
    J_T_fhss += A_J * np.cos(2 * np.pi * f_jam_ch * t + np.random.uniform(0, 2*np.pi))

y_fhss = x_fhss + J_T_fhss + J_N + J_S + n_t

decoded_fhss = np.zeros(K)
for k in range(K):
    start = k * N_sym
    end = (k + 1) * N_sym
    f_hop = f_min + hop_seq[k] * delta_f
    ref = np.cos(2 * np.pi * f_hop * t[start:end])
    M_k = np.sum(y_fhss[start:end] * ref)
    decoded_fhss[k] = 1 if M_k > 0 else 0

err_fhss = count_errors(d_k, decoded_fhss)
ber_fhss = err_fhss / K
eta = N_jam / N_ch

# ================= 8. МЕТОД 4: WATERMARKING + ЭШЕЛОНИРОВАНИЕ =================
alpha_w = 0.15 * A
np.random.seed(999)
w_bits = np.random.randint(0, 2, K)
w_symbols = 2 * w_bits - 1
w_modulated = np.repeat(w_symbols, N_sym)
w_t = w_modulated * np.sin(2 * np.pi * f0 * t)

# Сигнал с тегом
x_auth = x_t + alpha_w * w_t

# Спуфер формирует сигнал БЕЗ тега
x_spoof_no_tag = A_S * modulated_spoof * carrier_spoof
J_S_auth = np.zeros(N)
if tau_S_samples >= 0:
    J_S_auth[tau_S_samples:] = x_spoof_no_tag[:N-tau_S_samples]

# Принятая смесь (с тональной, шумовой и спуфинг-помехами)
y_auth_raw = x_auth + J_T + J_N + J_S_auth + n_t

# === ЭШЕЛОНИРОВАНИЕ: сначала режекция тональной помехи ===
Y_auth = np.fft.fft(y_auth_raw)
spec_auth = np.abs(Y_auth)
lambda_auth = np.mean(spec_auth) + beta * np.std(spec_auth)
mask_auth = np.ones(N, dtype=complex)
mask_auth[np.where(spec_auth > lambda_auth)] = 0
y_auth = np.fft.ifft(Y_auth * mask_auth).real  # Очищенный от тональной помехи сигнал

# Обнаружение тега (корреляция с известным w_t)
corr_w = np.sum(y_auth * w_t)
auto_w = np.sum(w_t * w_t)
threshold_w = 0.5 * alpha_w * auto_w
spoof_detected = corr_w < threshold_w

if spoof_detected:
    # Удаляем тег перед демодуляцией
    y_clean_auth = y_auth - alpha_w * w_t
    decoded_auth = np.zeros(K)
    for k in range(K):
        start = k * N_sym
        end = (k + 1) * N_sym
        M_k = np.sum(y_clean_auth[start:end] * np.cos(2*np.pi*f0*t[start:end]))
        decoded_auth[k] = 1 if M_k > 0 else 0
    err_auth = count_errors(d_k, decoded_auth)
    ber_auth = err_auth / K
else:
    err_auth = K
    ber_auth = 1.0
# ================= 9. СВОДНАЯ ТАБЛИЦА =================
print(f"\n{'='*60}")
print(f"СРАВНЕНИЕ МЕТОДОВ БОРЬБЫ С ПОМЕХАМИ")
print(f"{'='*60}")
print(f"DSSS (Баркер-11, G_p={Gp_dB:.1f} дБ): ошибок {err_dsss}/{K}, BER = {ber_dsss:.3f}")
print(f"FHSS  (N_ch={N_ch}, подавлено {N_jam} каналов, η={eta:.1f}): ошибок {err_fhss}/{K}, BER = {ber_fhss:.3f}")
print(f"Watermarking:  корр. R_w={corr_w:.1f}, порог λ_w={threshold_w:.1f}")
print(f"                спуфинг обнаружен: {'ДА' if spoof_detected else 'НЕТ'}")
print(f"                ошибок {err_auth}/{K}, BER = {ber_auth:.3f}")
print(f"\n{'='*60}")
print(f"СВОДНАЯ ТАБЛИЦА")
print(f"{'='*60}")
print(f"{'Метод':<25} {'BER':<10} {'Против':<30}")
print(f"{'-'*60}")
print(f"{'Без защиты':<25} {ber_jammed:<10.3f} {'—':<30}")
print(f"{'Частотная режекция':<25} {ber_filtered:<10.3f} {'Тональная J_T':<30}")
print(f"{'DSSS (Баркер-11)':<25} {ber_dsss:<10.3f} {'Шумовая J_N':<30}")
print(f"{'FHSS (ППРЧ)':<25} {ber_fhss:<10.3f} {'Тональная + шумовая':<30}")
print(f"{'Watermarking':<25} {ber_auth:<10.3f} {'Имитационная J_S':<30}")
print(f"{'='*60}")
print(f"Вывод: эшелонирование — режекция + DSSS/FHSS + аутентификация")
print(f"обеспечивает защиту от всех типов преднамеренных помех.")

# ================= 10. ВИЗУАЛИЗАЦИЯ (РАСШИРЕННАЯ) =================
fig, axs = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle('Сравнение методов борьбы с преднамеренными помехами',
             fontsize=16, fontweight='bold')

# --------------------- СТРОКА 1: РЕЖЕКЦИЯ ---------------------

# График 1.1: Спектр до режекции
axs[0, 0].plot(freqs[:N//2]/1e3, spectrum_mag[:N//2], 'r', alpha=0.8)
axs[0, 0].axhline(y=lambda_th, color='k', linestyle='--', linewidth=1.5,
                  label=f'Порог λ = {lambda_th:.0f}')
axs[0, 0].set_title('Режекция: спектр |Y(f)| до обработки')
axs[0, 0].set_xlabel('Частота, кГц')
axs[0, 0].set_ylabel('|Y(f)|')
axs[0, 0].legend(fontsize=8)
axs[0, 0].grid(True, alpha=0.3)

# График 1.2: Спектр после режекции
spec_filtered = np.abs(np.fft.fft(y_hat))
axs[0, 1].plot(freqs[:N//2]/1e3, spec_filtered[:N//2], 'b')
axs[0, 1].set_title('Режекция: спектр |Ŷ(f)| после обработки')
axs[0, 1].set_xlabel('Частота, кГц')
axs[0, 1].set_ylabel('|Ŷ(f)|')
axs[0, 1].grid(True, alpha=0.3)

# График 1.3: Сигналы во времени (режекция)
window = 150
axs[0, 2].plot(t[:window]*1e3, x_t[:window], 'g', linewidth=2, label='Исходный x(t)')
axs[0, 2].plot(t[:window]*1e3, y_t[:window], 'r', alpha=0.5, label='С помехами y(t)')
axs[0, 2].plot(t[:window]*1e3, y_hat[:window], 'b', linewidth=1.5, label='После режекции ŷ(t)')
axs[0, 2].set_title('Режекция: сигналы во времени')
axs[0, 2].set_xlabel('Время, мс')
axs[0, 2].set_ylabel('Амплитуда')
axs[0, 2].legend(fontsize=7)
axs[0, 2].grid(True, alpha=0.3)

# --------------------- СТРОКА 2: DSSS ---------------------

# График 2.1: Код Баркера
axs[1, 0].stem(range(L_c), barker_11, linefmt='b-', markerfmt='bo', basefmt=' ')
axs[1, 0].set_title(f'DSSS: код Баркера (L_c={L_c})')
axs[1, 0].set_xlabel('Номер чипа')
axs[1, 0].set_ylabel('c_n')
axs[1, 0].set_ylim(-1.5, 1.5)
axs[1, 0].grid(True, alpha=0.3, axis='y')

# График 2.2: Сравнение спектров (до/после сжатия DSSS)
spec_dsss_before = np.abs(np.fft.fft(y_dsss))
spec_dsss_after = np.abs(np.fft.fft(y_despread))
freqs_dsss = np.fft.fftfreq(N_dsss_total, 1/Fs)
axs[1, 1].plot(freqs_dsss[:N_dsss_total//2]/1e3, spec_dsss_before[:N_dsss_total//2],
               'r', alpha=0.6, label='До сжатия')
axs[1, 1].plot(freqs_dsss[:N_dsss_total//2]/1e3, spec_dsss_after[:N_dsss_total//2],
               'b', alpha=0.8, label='После сжатия')
axs[1, 1].set_title(f'DSSS: спектр (G_p = {Gp_dB:.1f} дБ)')
axs[1, 1].set_xlabel('Частота, кГц')
axs[1, 1].set_ylabel('Амплитуда')
axs[1, 1].legend(fontsize=8)
axs[1, 1].grid(True, alpha=0.3)

# График 2.3: Биты DSSS
axs[1, 2].stem(range(K), d_k, linefmt='g-', markerfmt='go', basefmt=' ',
               label='Передано')
axs[1, 2].stem(range(K), decoded_dsss + 0.1, linefmt='b-', markerfmt='bs', basefmt=' ',
               label=f'Принято (DSSS), BER={ber_dsss:.2f}')
axs[1, 2].set_title('DSSS: переданные и принятые биты')
axs[1, 2].set_xlabel('Номер бита k')
axs[1, 2].set_ylabel('Значение')
axs[1, 2].set_yticks([0, 1])
axs[1, 2].set_ylim(-0.3, 1.5)
axs[1, 2].legend(fontsize=8)
axs[1, 2].grid(True, alpha=0.3, axis='y')

# --------------------- СТРОКА 3: FHSS, WATERMARKING, СВОДКА ---------------------

# График 3.1: Частотный план FHSS
freq_plan = np.arange(N_ch) * delta_f + f_min
colors_fh = ['red' if ch in jammed_channels else 'blue' for ch in range(N_ch)]
axs[2, 0].bar(range(N_ch), np.ones(N_ch), color=colors_fh, edgecolor='black')
for ch in range(N_ch):
    axs[2, 0].text(ch, 0.5, f'{freq_plan[ch]/1e3:.0f}', ha='center', va='center', fontsize=8)
axs[2, 0].set_title(f'FHSS: частотный план (N_ch={N_ch}, подавлено {N_jam})')
axs[2, 0].set_xlabel('Номер канала')
axs[2, 0].set_ylabel('')
axs[2, 0].set_yticks([])
axs[2, 0].legend(handles=[
    plt.Rectangle((0,0), 1, 1, color='blue', label='Чистый'),
    plt.Rectangle((0,0), 1, 1, color='red', label='Подавлен')
], fontsize=8)

# График 3.2: Корреляция Watermarking
corr_per_bit = np.zeros(K)
for k in range(K):
    start = k * N_sym
    end = (k + 1) * N_sym
    corr_per_bit[k] = np.sum(y_auth[start:end] * w_t[start:end])

axs[2, 1].bar(range(K), corr_per_bit, color=['blue' if c > 0 else 'red' for c in corr_per_bit],
              edgecolor='black')
axs[2, 1].axhline(y=threshold_w/K, color='k', linestyle='--',
                  label=f'Порог (на бит) = {threshold_w/K:.0f}')
axs[2, 1].set_title(f'Watermarking: корреляция по битам\nСпуфинг: {"ОБНАРУЖЕН" if spoof_detected else "НЕ ОБНАР."}')
axs[2, 1].set_xlabel('Номер бита k')
axs[2, 1].set_ylabel('R_w (на бит)')
axs[2, 1].legend(fontsize=8)
axs[2, 1].grid(True, alpha=0.3, axis='y')

# График 3.3: Сводная гистограмма BER (все методы)
methods = ['Без\nзащиты', 'Режекция', 'DSSS', 'FHSS', 'Water-\nmarking']
ber_values_all = [ber_jammed, ber_filtered, ber_dsss, ber_fhss, ber_auth]
colors_bar = ['red', 'orange', 'blue', 'cyan', 'magenta']
bars = axs[2, 2].bar(methods, ber_values_all, color=colors_bar, edgecolor='black', width=0.6)
axs[2, 2].set_title('Сравнение BER по методам защиты')
axs[2, 2].set_ylabel('BER')
axs[2, 2].set_ylim(0, max(0.8, max(ber_values_all)*1.3))
for bar, val in zip(bars, ber_values_all):
    axs[2, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
# Добавим линию теоретического предела
axs[2, 2].axhline(y=ber_theory, color='gray', linestyle=':', linewidth=1.5,
                  label=f'Теория BPSK = {ber_theory:.4f}')
axs[2, 2].legend(fontsize=8)
axs[2, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('methods_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nГрафик сохранён в 'methods_comparison.png'")
