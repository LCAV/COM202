import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets


def tdplot(s, fs, name, plottitle):
    Ts = 1 / fs
    t = np.linspace(0, (np.size(s) - 1) * Ts, np.size(s))

    fig, axs = plt.subplots(constrained_layout=True)
    axs.plot(t, s) 
    axs.set(xlabel='t [s]', ylabel=name+'(t)')
    plt.show()

    return

def fdplot(s, fs, name, plottitle):
    Ts = 1 / fs
    t = np.linspace(0, (np.size(s) - 1) * Ts, np.size(s))

    NFFT = np.ceil(np.log2(np.size(s)))
    NFFT = np.power(2, NFFT)
    f = np.linspace(-1 / (2 * Ts), 1 / (2 * Ts) - 1 / (NFFT * Ts), int(NFFT))

    s_f = np.fft.fft(s, int(NFFT))
    s_f = Ts * np.fft.fftshift(s_f)

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle(plottitle, fontsize=16)
    if np.iscomplexobj(s):
        axs[0].plot(t, s.real) 
        axs[0].plot(t, s.imag)
    else:
        axs[0].plot(t, s)
    axs[0].set(xlabel='t [s]', ylabel=name+'(t)')
    axs[1].plot(f, np.abs(s_f))
    axs[1].set(xlabel='f [Hz]', ylabel='|'+name+'_F(f)|')
    plt.show()

    return s_f, f
    
def organize_tab(freq1_slider, amp1_slider, show_exp1_checkbox, freq2_slider, amp2_slider, show_exp2_checkbox, freq3_slider, amp3_slider, show_exp3_checkbox):
    # create the tab widget
    tabs = widgets.Tab(children=[
        widgets.VBox([freq1_slider, amp1_slider, show_exp1_checkbox]),
        widgets.VBox([freq2_slider, amp2_slider, show_exp2_checkbox]),
        widgets.VBox([freq3_slider, amp3_slider, show_exp3_checkbox])
    ])
    tabs.set_title(0, 'Exponential 1')
    tabs.set_title(1, 'Exponential 2')
    tabs.set_title(2, 'Exponential 3')
    return tabs

def create_sliders():
    # define the sliders and checkboxes for each complex exponential
    freq1_slider = widgets.FloatSlider(value=440, min=-2500, max=2500, step=10, description='Frequency 1:')
    amp1_slider = widgets.FloatSlider(value=1, min=0, max=5, step=1, description='Amplitude 1:')
    show_exp1_checkbox = widgets.Checkbox(value=True, description='Show Exponential 1')

    freq2_slider = widgets.FloatSlider(value=-440, min=-2500, max=2500, step=10, description='Frequency 2:')
    amp2_slider = widgets.FloatSlider(value=1, min=0, max=5, step=1, description='Amplitude 2:')
    show_exp2_checkbox = widgets.Checkbox(value=False, description='Show Exponential 2')

    freq3_slider = widgets.FloatSlider(value=1320, min=-2500, max=2500, step=10, description='Frequency 3:')
    amp3_slider = widgets.FloatSlider(value=1, min=0, max=5, step=1, description='Amplitude 3:')
    show_exp3_checkbox = widgets.Checkbox(value=False, description='Show Exponential 3')
    
    return freq1_slider, amp1_slider, show_exp1_checkbox, freq2_slider, amp2_slider, show_exp2_checkbox, freq3_slider, amp3_slider, show_exp3_checkbox