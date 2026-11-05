# Task

Your task is to edit the notes contained in /ceph/scratch/jlaherty/Projects/FunctionalCellTypes/latex

Without marking edits:
1. Ensure consistent notation. For example, all layer indexes should appear in superscript with parentheses, i.e. $\cdot^{(l)}$; all data indices $\mu,\nu$, etc. should appear in superscript, whereas all spatial indices $i,j$, $a,b$, $\alpha,\beta$ are subscripts.
Create macros for the following edits:
2. Correct trivial typos in prose and equations with green
3. Correct non-trivial but non-load-bearing mistakes with orange, and leave a comment nearby (also in orange) explaining the mistake, its consequences, and how to fix it
4. Correct larger/conceptual/load-bearing mistakes in red, then leave a self-contained latex note in latex/notes that discusses the problem in detail and how to address it
Other work to perform:
5. Create an appendix section with A1, A2, etc., numbering and put A1_comments.tex here
Analysis to perform with Fable workers:
6. at the end of 02_renormalisation_group.tex, there is eq:linear-soln. Analyse this solution for intuition in various limiting cases, in particular
$\sigma \rightarrow 0$, $g \rightarrow 0$, $k_0 = k_1 \equiv k$, specifying a particular spectrum for the $\lambda_q$s and where the dataset is generated as $X = Z W_X^{\top}, Y = Z W_Y^{\top}$ with $Z \in \mathbb{R}^{P \times K}$ being the 'latents': each is drawn i.i.d from a uniform distribution on $[0, a]$; $W_X$ and $W_Y$ having orthonormal columns. We are looking for conclusions about what features of this task lead to the development of 'functional cell types', which would be cells that correlate with particular latents, and the spatial scale at which these cells emerge. As part of this, leave a concrete set of conditions to test numerically to verify the theory. If your editing process yields substnatial changes to the conclusion eq:linear-soln, then include a section at the top of the document that addresses how your edits would change the current form \eq:linear-soln solution, and then perform the analysis. Put your work in latex/notes as a self-contained latex document.
7. Your analysis in (6) should result in a concrete set of numerical experiments to run. Consider the fct codebase and evaluate whether there are any discrepancies between the model architecture and conventions as they exist in the latex and the codebase. Describe the discrepancies and how they should be addressed in the code (we want the theory to lead the code conventions). Conclude with how the experiments should be run using fct. Give absolute detail, leaving a note in latex/notes. This will be taken up by a subsequent session to implement the changes.
8. Similarly, leave a note (in a different file to point 7) discussing the possibility of solving the linear case without making the mutual-diagonalisability ansatz: $\Sigma_q = U D_q U^{\top}, X = U S_X V_X^{\top}, Y = U S_Y V^{\top}$
9. Leave a note showing the translation of this discrete analysis into the continuous field limit (as $N \rightarrow \infty$, the lattice spacing of the system should stay fixed while the volume increases; consequently, introduce a maximum fourier scale $\Lambda$)