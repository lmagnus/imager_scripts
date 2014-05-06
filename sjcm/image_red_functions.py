#
# Script that produced first KAT-7 image.
#
# Ludwig Schwardt
# 18 July 2011
#

import os
import time
import optparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import ndimage

import katfile
import katpoint
import scape

def plot_vis_crosshairs(fig, vis_data, title, crosscorr, ants, inputs, upper=True, units='', **kwargs):
    """Create phasor plot (upper or lower triangle of baseline matrix)."""
    fig.subplots_adjust(wspace=0., hspace=0.)
    data_lim = np.max([np.abs(vis).max() for vis in vis_data])
    ax_lim = 1.05 * data_lim
    for n, (indexA, indexB) in enumerate(crosscorr):
        subplot_index = (len(ants) * indexA + indexB + 1) if upper else (indexA + len(ants) * indexB + 1)
        ax = fig.add_subplot(len(ants), len(ants), subplot_index)
        for vis in vis_data:
            ax.plot(vis[:, n].real, vis[:, n].imag, **kwargs)
        ax.axhline(0, lw=0.5, color='k')
        ax.axvline(0, lw=0.5, color='k')
        ax.add_patch(mpl.patches.Circle((0., 0.), data_lim, facecolor='none', edgecolor='k', lw=0.5))
        ax.add_patch(mpl.patches.Circle((0., 0.), 0.5 * data_lim, facecolor='none', edgecolor='k', lw=0.5))
        ax.axis('image')
        ax.axis([-ax_lim, ax_lim, -ax_lim, ax_lim])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        if upper:
            if indexA == 0:
                ax.xaxis.set_label_position('top')
                ax.set_xlabel(inputs[indexB][3:])
            if indexB == len(ants) - 1:
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(inputs[indexA][3:], rotation='horizontal')
        else:
            if indexA == 0:
                ax.set_ylabel(inputs[indexB][3:], rotation='horizontal')
            if indexB == len(ants) - 1:
                ax.set_xlabel(inputs[indexA][3:])
    fig.text(0.5, 0.95 if upper else 0.05, title, ha='center', va='bottom' if upper else 'top')
    fig.text(0.95 if upper else 0.05, 0.5, 'Outer radius = %g %s' % (data_lim, units), va='center', rotation='vertical')


def nanmean(array_data, axis=0):
    """Calculates the mean of the array ignoring the nan values """

    mdat   = np.ma.masked_array(array_data, np.isnan(array_data));
    retval = np.mean(mdat, axis=axis);
 
    return retval;

def nanvar(array_data, axis=0):
    """Calculates the mean of the array ignoring the nan values """

    mdat   = np.ma.masked_array(array_data, np.isnan(array_data));
    retval = np.var(mdat, axis=axis);
 
    return retval;


def nanvarc(array_data, axis=0):
    """Calculates the complex variance the array ignoring the nan values """

    mdatreal = np.ma.masked_array(array_data.real, np.isnan(array_data.real));
    varreal  = np.var(mdatreal, axis=axis);
    mdatimag = np.ma.masked_array(array_data.imag, np.isnan(array_data.imag));
    varimag  = np.var(mdatimag, axis=axis);
    retval   = np.array( (varreal + 1j*varimag) );
 
    return retval;


def nansum(array_data, axis=0):
    """Calculates the sum of the array ignoring the nan values """

    mdat   = np.ma.masked_array(array_data, np.isnan(array_data));
    retval = np.sum(mdat, axis=axis);
 
    return retval;


def apply_gains(params, input_pairs, model_vis=1.0):
    """Apply relevant antenna gains to model visibility to estimate measurements.

    This corrupts the ideal model visibilities by applying a set of complex
    antenna gains to them.

    Parameters
    ----------
    params : array of float, shape (2 * N - 1,)
        Array of gain parameters with 2 parameters per signal path (real and
        imaginary components of complex gain), except for phase reference input
        which has a single real component
    input_pairs : array of int, shape (2, M)
        Input pair(s) for which visibilities will be calculated. The inputs are
        specified by integer indices into the main input list.
    model_vis : complex or array of complex, shape (M,), optional
        The modelled (ideal) source visibilities on each specified baseline.
        The default model is that of a point source with unit flux density.

    Returns
    -------
    estm_vis : array of float, shape (2,) or (2, M)
        Estimated visibilities specified by their real and imaginary components

    """
    full_params[params_to_fit] = params
    antA, antB = input_pairs[0], input_pairs[1]
    reA, imA, reB, imB = full_params[2 * antA], full_params[2 * antA + 1], full_params[2 * antB], full_params[2 * antB + 1]
    # Calculate gain product (g_A g_B*)
    reAB, imAB = reA * reB + imA * imB, imA * reB - reA * imB
    re_model, im_model = np.real(model_vis), np.imag(model_vis)
    return np.vstack((reAB * re_model - imAB * im_model, reAB * im_model + imAB * re_model)).squeeze()



def apply_phases(params, input_pairs, model_vis=1.0):
    """Apply relevant antenna phases to model visibility to estimate measurements.

    This corrupts the ideal model visibilities by adding a set of per-antenna
    phases to them.

    Parameters
    ----------
    params : array of float, shape (N - 1,)
        Array of gain parameters with 1 parameter per signal path (phase
        component of complex gain), except for phase reference input which has
        zero phase
    input_pairs : array of int, shape (2, M)
        Input pair(s) for which visibilities will be calculated. The inputs are
        specified by integer indices into the main input list.
    model_vis : complex or array of complex, shape (M,), optional
        The modelled (ideal) source visibilities on each specified baseline.
        The default model is that of a point source with unit flux density.

    Returns
    -------
    estm_vis : array of float, shape (2,) or (2, M)
        Estimated visibilities specified by their real and imaginary components

    """
    phase_params[phase_params_to_fit] = params
    phaseA, phaseB = phase_params[input_pairs[0]], phase_params[input_pairs[1]]
    # Calculate gain product (g_A g_B*) where each gain has unit magnitude
    estm_vis = np.exp(1j * (phaseA - phaseB)) * model_vis
    return np.vstack((np.real(estm_vis), np.imag(estm_vis))).squeeze()


# The CLEAN variant(s) that will be used
def omp(A, y, S, At_times=None, A_column=None, N=None, printEveryIter=1, resThresh=0.0):
    """Orthogonal Matching Pursuit.
    
    This approximately solves the linear system A x = y for sparse x, where A is
    an MxN matrix with M << N.
    
    Parameters
    ----------
    A : array, shape (M, N)
        The measurement matrix of compressed sensing (None for implicit functions)
    y : array, shape (M,)
        A vector of measurements
    S : integer
        Maximum number of sparse components to find (sparsity level), between 1 and M.
    At_times : function
        Function that calculates At_times(x) = A' * x implicitly. It takes
        an array of shape (M,) as argument and returns an array of shape (N,).
        Default is None.
    A_column : function
        Function that returns the n-th column of A as A_column(n) = A[:, n]. It takes
        an integer as argument and returns an array of shape (M,). Default is None.
    N : integer
        Number of columns in matrix A (dictionary size in MP-speak). Default is None,
        which means it is automatically determined from A.
    printEveryIter : integer
        A progress line is printed every 'printEveryIter' iterations (0 for no
        progress report). The default is a progress report after every iteration.
    resThresh : real
        Stop iterating if the residual l2-norm (relative to the l2-norm of the 
        measurements y) falls below this threshold. Default is 0.0 (no threshold).

    Returns
    -------
    x : array of same type as y, shape (N,)
        The approximate sparse solution to A x = y.
    
    """
    # Convert explicit A matrix to functional form (or use provided functions)
    if At_times is None:
        At_times = lambda x: np.dot(A.conjugate().transpose(), x) 
        A_column = lambda n: A[:, n]
        N = A.shape[1]
    M = len(y)
    # (1) -- Initialization
    residual = y
    resSize = 1.0
    atoms = np.zeros((M, S), dtype=y.dtype)
    atomIndex = np.zeros(S, dtype='int32')
    # (6) -- Loop until the desired number of atoms are found (may stop before that)
    for s in xrange(S):
        # (2) -- Easy optimization problem (peakpicking the dirty residual image |A' r|)
        atomIndex[s] = np.abs(At_times(residual)).argmax()
        # Stop if this atom has already been included in active set (if OMP revisits an
        # atom, it can only be due to round-off error), or if residual threshold is crossed
        if (atomIndex[s] in atomIndex[:s]) or (resSize < resThresh):
            atomIndex = atomIndex[:s]
            break
        # (3) -- Update matrix of atoms
        atoms[:, s] = A_column(atomIndex[s])
        activeAtoms = atoms[:, :s+1]
        # (4) -- Solve least-squares problem to obtain new signal estimate
        atomWeights = np.linalg.lstsq(activeAtoms, y)[0].real
        # (5) -- Calculate new residual
        residual = y - np.dot(activeAtoms, atomWeights)
        resSize = np.linalg.norm(residual) / np.linalg.norm(y)
        if printEveryIter and ((s+1) % printEveryIter == 0):
            print "%d : atom = %d, dual = %.3e, residual l2 = %.3e" % \
                  (s, atomIndex[s], atomWeights[s], resSize)
    # (7) -- Create signal estimate
    x = np.zeros(N, dtype=y.dtype)
    x[atomIndex] = atomWeights[:len(atomIndex)]
    if printEveryIter:
        print 'omp: atoms = %d, residual = %.3e' % (sum(x != 0.0), resSize)
    return x

def omp_plus(A, y, S, At_times=None, A_column=None, N=None, printEveryIter=1, resThresh=0.0):
    """Positive Orthogonal Matching Pursuit.

    This approximately solves the linear system A x = y for sparse positive real x,
    where A is an MxN matrix with M << N. This is very similar to the NNLS algorithm of
    Lawson & Hanson (Solving Least Squares Problems, 1974, Chapter 23).

    Parameters
    ----------
    A : array, shape (M, N)
        The measurement matrix of compressed sensing (None for implicit functions)
    y : array, shape (M,)
        A vector of measurements
    S : integer
        Maximum number of sparse components to find (sparsity level), between 1 and M.
    At_times : function
        Function that calculates At_times(x) = A' * x implicitly. It takes
        an array of shape (M,) as argument and returns an array of shape (N,).
        Default is None.
    A_column : function
        Function that returns the n-th column of A as A_column(n) = A[:, n]. It takes
        an integer as argument and returns an array of shape (M,). Default is None.
    N : integer
        Number of columns in matrix A (dictionary size in MP-speak). Default is None,
        which means it is automatically determined from A.
    printEveryIter : integer
        A progress line is printed every 'printEveryIter' iterations (0 for no
        progress report). The default is a progress report after every iteration.
    resThresh : real
        Stop iterating if the residual l2-norm (relative to the l2-norm of the
        measurements y) falls below this threshold. Default is 0.0 (no threshold).

    Returns
    -------
    x : array of same type as y, shape (N,)
        The approximate sparse positive solution to A x = y.

    """
    # Convert explicit A matrix to functional form (or use provided functions)
    if At_times is None:
        At_times = lambda x: np.dot(A.conjugate().transpose(), x)
        A_column = lambda n: A[:, n]
        N = A.shape[1]
    M = len(y)
    # Initialization
    residual = y
    resSize = 1.0
    atoms = np.zeros((M, S), dtype=y.dtype)
    atomIndex = -np.ones(S, dtype='int32')
    atomWeights = np.zeros(S, dtype='float64')
    atomHistory = set()
    numAtoms = 0
    iterCount = 0
    # Maximum iteration count suggested by Lawson & Hanson
    iterMax = 3 * N

    try:
        # MAIN LOOP (a la NNLS) to find all atoms
        # Loop until the desired number of components / atoms are found, or the
        # residual size drops below the threshold (an earlier exit is also possible)
        while (numAtoms < S) and (resSize >= resThresh):
            # Form the real part of the dirty image residual A' r
            # This happens to be the negative gradient of 0.5 || y - A x ||_2^2,
            # the associated l2-norm (least-squares) objective, and also the dual vector
            dual = At_times(residual).real
            # Ensure that no existing atoms will be selected again
            dual[atomIndex[:numAtoms]] = 0.0
            # Loop until a new atom with positive weight is found, or die trying
            while True:
                newAtom = dual.argmax()
                # Stop if atom is already in active set, or gradient is non-positive
                if dual[newAtom] <= 0.0:
                    break
                # Tentatively add new atom to active set
                atomIndex[numAtoms] = newAtom
                atoms[:, numAtoms] = A_column(newAtom)
                activeAtoms = atoms[:, :numAtoms+1]
                # Solve unconstrained least-squares problem (Gram-Schmidt orthogonalisation step)
                newWeights = np.linalg.lstsq(activeAtoms, y)[0].real
                # If weight of new atom is non-positive, discard it and go for next best atom
                if newWeights[-1] <= 0.0:
                    dual[newAtom] = 0.0
                else:
                    break
            if dual[newAtom] <= 0.0:
                break
            # If search has been in this state before, it will get stuck in endless loop
            # until iterMax is reached, which is pointless
            # TODO: check the effort involved in this check (maybe we don't need it if the
            # endless loop is due to a bug somewhere else?)
            atomState = tuple(sorted(atomIndex[:numAtoms+1]))
            if atomState in atomHistory:
                print "endless loop detected, terminating"
                break
            else:
                atomHistory.add(atomState)
            numAtoms += 1
            # SECONDARY LOOP (a la NNLS) to get all atom weights to be positive simultaneously
            # Forced to terminate if it takes too long
            while iterCount <= iterMax:
                iterCount += 1
                # Check for non-positive weights
                nonPos = [n for n in xrange(len(newWeights)) if newWeights[n] <= 0.0]
                if len(nonPos) == 0:
                    break
                # Interpolate between old and new weights so that at least one atom
                # with negative weight now has zero weight, and can therefore be discarded
                oldWeights = atomWeights[:numAtoms]
                alpha = oldWeights[nonPos] / (oldWeights[nonPos] - newWeights[nonPos])
                worst = alpha.argmin()
                oldWeights += alpha[worst] * (newWeights - oldWeights)
                # Make sure the selected atom really has 0 weight (round-off could change it)
                oldWeights[nonPos[worst]] = 0.0
                # Only keep the atoms with positive weights (could be more efficient...)
                goodAtoms = [n for n in xrange(len(oldWeights)) if oldWeights[n] > 0.0]
                numAtoms = len(goodAtoms)
                print "iter %d : best atom = %d, found negative weights, worst at %d, reduced atoms to %d" % \
                      (iterCount, newAtom, atomIndex[nonPos[worst]], numAtoms)
                atomIndex[:numAtoms] = atomIndex[goodAtoms].copy()
                atomIndex[numAtoms:] = -1
                activeAtoms = atoms[:, goodAtoms].copy()
                atoms[:, :numAtoms] = activeAtoms
                atoms[:, numAtoms:] = 0.0
                atomWeights[:numAtoms] = atomWeights[goodAtoms].copy()
                atomWeights[numAtoms:] = 0.0
                # Solve least-squares problem again to get new proposed atom weights
                newWeights = np.linalg.lstsq(activeAtoms, y)[0].real
            if iterCount > iterMax:
                break
            # Accept new weights, update residual and continue with main loop
            atomWeights[:numAtoms] = newWeights
            residual = y - np.dot(activeAtoms, newWeights)
            resSize = np.linalg.norm(residual) / np.linalg.norm(y)
            if printEveryIter and (iterCount % printEveryIter == 0):
                print "iter %d : best atom = %d, dual = %.3e, atoms = %d, residual l2 = %.3e" % \
                      (iterCount, newAtom, dual[newAtom], numAtoms, resSize)

    # Return last results on Ctrl-C, for the impatient ones
    except KeyboardInterrupt:
        # Create sparse solution vector
        x = np.zeros(N, dtype='float64')
        x[atomIndex[:numAtoms]] = atomWeights[:numAtoms]
        if printEveryIter:
            print 'omp: atoms = %d, residual = %.3e (interrupted)' % (sum(x != 0.0), resSize)

    else:
        # Create sparse solution vector
        x = np.zeros(N, dtype='float64')
        x[atomIndex[:numAtoms]] = atomWeights[:numAtoms]
        if printEveryIter:
            print 'omp: atoms = %d, residual = %.3e' % (sum(x != 0.0), resSize)

    return x

################################# SELF-CAL #####################################

# This follows the recipe in Chapter 10 of the White Book

# Step 1: Make an initial model of the source (we have the model_vis_samples obtained by the initial CLEAN)

# Step 2: Convert the source into a point source using the model (this happens inside the solver)

# Step 3: Solve for the time-varying complex antenna gains
# selfcal_type = 'P'
# selfcal_gains = []
# uv_dist_range = [0, 1500]
# bins_per_solint = 1
# input_pairs = np.tile(np.array(crosscorr).T, len(start_chans))
# solint_size = len(start_chans) * len(crosscorr) * bins_per_solint
# # Iterate over solution intervals
# for n in range(0, len(vis_samples), solint_size):
#     vis, model_vis, uvd = vis_samples[n:n + solint_size], model_vis_samples[n:n + solint_size], uvdist[n:n + solint_size]
#     good_uv = (uvd >= uv_dist_range[0]) & (uvd <= uv_dist_range[1])
#     if selfcal_type == 'P':
#         fitter = scape.fitting.NonLinearLeastSquaresFit(lambda p, x: apply_phases(p, x, model_vis[good_uv]), initial_phases)
#         fitter.fit(np.tile(input_pairs, bins_per_solint)[:, good_uv], np.vstack((vis.real, vis.imag))[:, good_uv])
#         phase_params[phase_params_to_fit] = fitter.params
#         gainsol = np.exp(1j * phase_params).astype(np.complex64)
#     else:
#         fitter = scape.fitting.NonLinearLeastSquaresFit(lambda p, x: apply_gains(p, x, model_vis[good_uv]), initial_gains)
#         fitter.fit(np.tile(input_pairs, bins_per_solint)[:, good_uv], np.vstack((vis.real, vis.imag))[:, good_uv])
#         full_params[params_to_fit] = fitter.params * np.sign(fitter.params[2 * ref_input_index])
#         gainsol = full_params.view(np.complex128).astype(np.complex64)
#     selfcal_gains.append(np.tile(gainsol, bins_per_solint).reshape(bins_per_solint, -1))
# # Solved gains per input and time bin
# selfcal_gains = np.vstack(selfcal_gains).T
# 
# fig = plt.figure(16)
# fig.clear()
# fig.subplots_adjust(right=0.8)
# ax = fig.add_subplot(121)
# for n in range(len(inputs)):
#     ax.plot(np.abs(selfcal_gains[n]), 'o', label=inputs[n][3:])
# ax.set_xlabel('Solution intervals')
# ax.set_title('Gain amplitude')
# ax = fig.add_subplot(122)
# for n in range(len(inputs)):
#     ax.plot(katpoint.rad2deg(np.angle(selfcal_gains[n])), 'o', label=inputs[n][3:])
# ax.set_xlabel('Solution intervals')
# ax.set_title('Gain phase (degrees)')
# ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0., numpoints=1)
# 
# # Step 4: Find the corrected visibility
# gainA = selfcal_gains.T[:, input_pairs[0, :]].ravel()
# gainB = selfcal_gains.T[:, input_pairs[1, :]].ravel()
# corrected_vis_samples = vis_samples / (gainA * gainB.conjugate())
# 
# # Step 5: Form a new model from the corrected data
# masked_comps = omp_plus(A=masked_phi, y=corrected_vis_samples, S=num_components, resThresh=res_thresh)
# model_vis_samples = np.dot(masked_phi, masked_comps)
# residual_vis = vis_samples - model_vis_samples
# print residual_vis.std()
# 
# # Step 6: Rinse back to step 2, repeat...
