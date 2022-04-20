Put your answer files here.

Remember to put your README.username in this directory as well.

TO DO:

1) Improve retrofitting algorithm to use all edges in the similarity graph. Use setMembership dict.
2) Tune the alpha and beta parameters and number of iterations T.
    a. Constrain alpha = Sum(beta_ij) for all i, j. Test for effectiveness.
    b. Constrain Sum(beta_i0j) for all j = Sum(beta_i1j) for all j, for any pair of words i0, i1. Test for effectiveness.
    c. Try increasing T while decreasing step size.
3) Use different lexicons.
4) Try context based methods.