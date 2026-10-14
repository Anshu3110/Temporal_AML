import os
import subprocess
from pathlib import Path

cur_dir = Path(__file__).resolve().parent

tex_3_5 = r"""\documentclass[border=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning}
\usepackage{amsmath, amssymb}
\begin{document}
\begin{tikzpicture}[
    tnode/.style={circle, draw=black, fill=blue!12, thick, minimum size=1.0cm, font=\small},
    inode/.style={circle, draw=red!80!black, fill=red!15, thick, minimum size=1.0cm, font=\small},
    edge/.style={->, >=stealth, thick, draw=gray!70},
    iedge/.style={->, >=stealth, thick, draw=red!60},
    tlabel/.style={font=\scriptsize, text=gray!85!black, fill=white, inner sep=1.5pt, rounded corners=1pt},
]
\node[tnode, label=left:{\tiny $t=5$}] (n1) at (0, 0) {$v_1$};
\node[tnode, label=above:{\tiny $t=7$}] (n2) at (2.5, 1.2) {$v_2$};
\node[inode, label=above:{\tiny $t=8$}] (n3) at (5.0, 1.2) {$v_3$};
\node[tnode, label=below:{\tiny $t=9$}] (n4) at (2.5,-1.2) {$v_4$};
\node[inode, label=right:{\tiny $t=12$}] (n5) at (7.5, 0) {$v_5$};

\draw[edge] (n1) -- (n2) node[midway, above, tlabel] {$\Delta t=2$};
\draw[edge] (n1) -- (n4) node[midway, below, tlabel] {$\Delta t=4$};
\draw[iedge] (n2) -- (n3) node[midway, above, tlabel] {$\Delta t=1$};
\draw[iedge] (n3) -- (n5) node[midway, above, tlabel] {$\Delta t=4$};
\draw[edge] (n4) -- (n5) node[midway, below, tlabel] {$\Delta t=3$};

% Legend
\node[tnode, minimum size=0.6cm, right=0.5cm of n5, yshift=0.5cm] (ll) {};
\node[right=0.1cm of ll, font=\tiny] {Licit};
\node[inode, minimum size=0.6cm, right=0.5cm of n5, yshift=-0.5cm] (li) {};
\node[right=0.1cm of li, font=\tiny] {Illicit};
\end{tikzpicture}
\end{document}
"""

tex_3_6 = r"""\documentclass[border=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning}
\usepackage{amsmath, amssymb}
\begin{document}
\begin{tikzpicture}[
    block/.style  = {rectangle, draw=blue!60!black,   fill=blue!8,    rounded corners=3pt, thick,
                     text width=5.2cm, minimum height=0.85cm, align=center, font=\small},
    param/.style  = {rectangle, draw=orange!80!black, fill=orange!12, rounded corners=3pt, thick,
                     text width=4.8cm, minimum height=0.85cm, align=center, font=\small},
    concat/.style = {rectangle, draw=green!60!black,  fill=green!8,   rounded corners=3pt, thick,
                     text width=5.2cm, minimum height=0.85cm, align=center, font=\small},
    normbox/.style = {rectangle, draw=green!60!black,  fill=green!15,  rounded corners=3pt, thick,
                      text width=5.2cm, minimum height=0.85cm, align=center, font=\small},
    arrow/.style  = {->, >=stealth, thick, draw=gray!60},
]
\node[block] (dt) at (0, 0) {Time Delta $\Delta t = t_i - t_j$};

\node[param]  (omega) at (-3.2, -1.6) {Learnable $\boldsymbol{\omega} \in \mathbb{R}^{64}$\\(Frequencies)};
\node[param]  (phi)   at ( 3.2, -1.6) {Learnable $\boldsymbol{\phi} \in \mathbb{R}^{64}$\\(Phases)};

\node[block]  (cos) at (-3.2, -3.4) {$\cos(\boldsymbol{\omega}\,\Delta t + \boldsymbol{\phi})$};
\node[block]  (sin) at ( 3.2, -3.4) {$\sin(\boldsymbol{\omega}\,\Delta t + \boldsymbol{\phi})$};

\node[concat] (cat)  at ( 0,   -5.0) {Concatenate: $[\cos \;\|\; \sin] \in \mathbb{R}^{128}$};

\node[normbox] (norm) at ( 0,   -6.6) {Scale $\times\!\sqrt{1/64}$ $\;\Rightarrow\;$ $\Phi(\Delta t)\!\in\!\mathbb{R}^{128}$};

\draw[arrow] (dt.south) -- ++(0,-0.3) -| (omega.north);
\draw[arrow] (dt.south) -- ++(0,-0.3) -| (phi.north);
\draw[arrow] (omega.south) -- (cos.north);
\draw[arrow] (phi.south) -- (sin.north);
\draw[arrow] (dt.south) -- ++(0,-0.3) -| ([xshift=0.8cm]cos.north);
\draw[arrow] (dt.south) -- ++(0,-0.3) -| ([xshift=-0.8cm]sin.north);
\draw[arrow] (cos.south) |- (cat.west);
\draw[arrow] (sin.south) |- (cat.east);
\draw[arrow] (cat.south) -- (norm.north);
\end{tikzpicture}
\end{document}
"""

(cur_dir / 'fig_3_5.tex').write_text(tex_3_5)
(cur_dir / 'fig_3_6.tex').write_text(tex_3_6)

env = os.environ.copy()
env['PATH'] = '/Library/TeX/texbin:/usr/bin:/bin'

subprocess.run(['pdflatex', '-interaction=nonstopmode', 'fig_3_5.tex'], cwd=cur_dir, env=env, check=True)
subprocess.run(['pdflatex', '-interaction=nonstopmode', 'fig_3_6.tex'], cwd=cur_dir, env=env, check=True)

subprocess.run(['sips', '-s', 'format', 'png', '-s', 'dpiWidth', '300', '-s', 'dpiHeight', '300', 'fig_3_5.pdf', '--out', 'figure_3_5.png'], cwd=cur_dir, check=True)
subprocess.run(['sips', '-s', 'format', 'png', '-s', 'dpiWidth', '300', '-s', 'dpiHeight', '300', 'fig_3_6.pdf', '--out', 'figure_3_6.png'], cwd=cur_dir, check=True)

print('Generated figure_3_5.png and figure_3_6.png successfully!')
