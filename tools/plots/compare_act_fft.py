import sys
sys.path.append('./')
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt





# # make a sine wave of fixed frequency
# signal_freqs = [10, 100, 1000]
# t = np.linspace(0, 1, 4000, endpoint=False)

# a = np.zeros_like(t)
# for sf in signal_freqs:
#     a += np.sin(2 * np.pi * sf * t)

# # quantize the sine to only update at freq Hz
# update_freq_for_quantization = 30
# time_period = 1/update_freq_for_quantization
# update_every = int(time_period*2000)
# quantized_a = np.zeros_like(a)
# for i in range(
#                 update_every, 
#                 len(a), 
#                 update_every,
#                 ):
#     quantized_a[i-update_every:i] = a[i-update_every]


# # perform fft
# a_fft = fft.fft(a)
# quantized_a_fft = fft.fft(quantized_a)
# fft_freq = fft.fftfreq(len(a), d=1.0/2000)

# # plot
# fig, axs = plt.subplots(1,2,figsize=(10,5))
# axs[0].plot(t, a)
# axs[0].plot(t, quantized_a)
# axs[0].set_xlabel('time (s)')
# axs[0].set_ylabel('amplitude')
# axs[0].grid()

# axs[1].plot(fft_freq, np.abs(a_fft))
# axs[1].plot(fft_freq, np.abs(quantized_a_fft))
# axs[1].set_xlim([0, 1000])
# axs[1].set_xlabel('frequency (Hz)')
# axs[1].set_ylabel('amplitude ')
# axs[1].grid()

# plt.tight_layout()
# plt.show()



# exit()










paths2logs = [
                'results/mr_f7_nm/9/log.npz',
                'results/mr_f9_nm2/0/log.npz',
                'results/mr_f10_nm/2/log.npz',
                'results/gfwg1/16/log.npz',

                # 'results/fgb_vary_obs/4/log.npz',
                # 'results/fgb_vary_obs/5/log.npz',

            ]

names = [
            'no_smooth',
            'gait_reward',
            'moving_avg_filter',
            'gait_reward2',
            # 'w/ clock',
            # 'w/o clock',
        ]


fig, axs = plt.subplots(2,5,figsize=(20,10))

for pi,path in enumerate(paths2logs):
    log = np.load(path,allow_pickle=True)
    actions = log['action']
    
    for i in range(actions.shape[1]):
        r = i // 5
        c = i % 5

        # perform fft
        action_fft = fft.fft(actions[:,i])
        fft_freq = fft.fftfreq(len(actions[:,i]), d=1.0/2000)

        # plot 
        
        axs[r,c].plot(fft_freq, np.abs(action_fft),  alpha=0.5, label=names[pi])

for i in range(actions.shape[1]):
    r = i // 5
    c = i % 5
    axs[r,c].grid()
    axs[r,c].legend(loc='upper right')
    axs[r,c].set_xlim([0, 100])
    axs[r,c].set_xlabel('frequency (Hz)')
    axs[r,c].set_ylim([0, 400])
    axs[r,c].set_ylabel('amplitude ')

    
    # axs[r,c].set_yscale('log')

    for max_amp_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        axs[r,c].axhline(y=max_amp_ratio*np.max(np.abs(action_fft)), color='r', linestyle='--', alpha=0.5)

    for freq in [30, 60, 90, 120]:
        axs[r,c].axvline(x=freq, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()