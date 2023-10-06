import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets

NUM_EXP = 4 # number of complex exponentials - you can change this
ctrls = { f'w{n}': widgets.FloatSlider(value=0.01, min=-0.2, max=0.2, step=0.01, description=f'$\\omega_{n}/(2\pi)$') for n in range(0, NUM_EXP) }
ctrls.update({ f'a{n}': widgets.FloatSlider(value=(1 if n == 0 else 0), min=0, max=5, step=1, description=f'$\\alpha_{n}$') for n in range(0, NUM_EXP) })
ctrls.update({ f's{n}': widgets.Checkbox(value=True, description='show') for n in range(0, NUM_EXP) })

tabs = widgets.Tab(children=[widgets.VBox([ctrls[f'w{n}'], ctrls[f'a{n}'], ctrls[f's{n}'] ]) for n in range(0, NUM_EXP)])
for n in range(0, NUM_EXP):
    tabs.set_title(n, f'cexp {n}')
    
def plot_re_im(axs, x, color='C0'):
    for n, s in enumerate([np.real(x), np.imag(x)]):
        axs[n].plot(s, color);
    
def plot_signals(C, N, figs, **kwargs):
    y = np.zeros(N, dtype=complex)
    for n in range(0, C):
        c = kwargs[f'a{n}'] * np.exp(2j * np.pi * kwargs[f'w{n}'] * np.arange(0, N))
        y += c
        if kwargs[f's{n}']:
            plot_re_im(figs, c, f'C{n+1}:');
    plot_re_im(figs, y);
    
N = 100 # signal length
def interactive_plot(**kwargs):
    fig = plt.figure();
    axs = [fig.add_subplot(1,2,1), fig.add_subplot(1, 2, 2)];
    fig.tight_layout(pad=2);
    axs[0].title.set_text('real part')
    axs[1].title.set_text('imaginary part')
    plot_signals(NUM_EXP, N, axs, **kwargs)
    
widget = widgets.VBox([tabs, widgets.interactive_output(interactive_plot, ctrls)])
display(widget);