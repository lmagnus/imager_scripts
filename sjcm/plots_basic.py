"""Basic plotting routines, used to create canned plots at a higher level."""

import logging

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    pass

try:
    import pyfits
except ImportError:
    pass

logger = logging.getLogger("scape.plots_basic")

def ordinal_suffix(n):
    """Returns the ordinal suffix of integer *n* as a string."""
    if n % 100 in [11, 12, 13]:
        return 'th'
    else:
        return {1 : 'st', 2 : 'nd', 3 : 'rd'}.get(n % 10, 'th')

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_line_segments
#--------------------------------------------------------------------------------------------------

def plot_line_segments(segments, labels=None, width=0.0, compact=True, add_breaks=True,
                       monotonic_axis='x', ax=None, **kwargs):
    """Plot sequence of line segments.

    This plots a sequence of line segments (of possibly varying length) on a
    single set of axes. Usually, one of the axes is considered *monotonic*,
    which means that the line segment coordinates along that axis increase
    monotonically through the sequence of segments. The classic example of such
    a plot is when the line segments represent time series data with time on the
    monotonic x-axis.

    Each segment may be labelled by a text string next to it. If *compact* is
    True, there will be no gaps between the segments along the monotonic axis.
    The tick labels on this axis are modified to reflect the original (padded)
    values. If *add_breaks* is True, the breaks between segments along the
    monotonic axis are indicated by dashed lines. If there is no monotonic axis,
    all these features (text labels, compaction and break lines) are disabled.

    Parameters
    ----------
    segments : sequence of array-like, shape (N_k, 2)
        Sequence of line segments (*line0*, *line1*, ..., *linek*, ..., *lineK*),
        where the k'th line is given by::

            linek = (x0, y0), (x1, y1), ... (x_{N_k - 1}, y_{N_k - 1})

        or the equivalent numpy array with two columns (for *x* and *y* values,
        respectively). Each line segment can be a different length. This is
        identical to the *segments* parameter of
        :class:`matplotlib.collections.LineCollection`.
    labels : sequence of strings, optional
        Corresponding sequence of text labels to add next to each segment along
        monotonic axis (only makes sense if there is such an axis)
    width : float, optional
        If non-zero, replace contiguous line with staircase levels of specified
        width along x-axis
    compact : {True, False}, optional
        Plot with no gaps between segments along monotonic axis (only makes
        sense if there is such an axis)
    add_breaks : {True, False}, optional
        Add vertical (or horizontal) lines to indicate breaks between segments
        along monotonic axis (only makes sense if there is such an axis)
    monotonic_axis : {'x', 'y', None}, optional
        Monotonic axis, along which segment coordinate increases monotonically
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)
    kwargs : dict, optional
        Extra keyword arguments are passed on to line collection constructor

    Returns
    -------
    segment_lines : :class:`matplotlib.collections.LineCollection` object
        Collection of segment lines
    break_lines : :class:`matplotlib.collections.LineCollection` object, or None
        Collection of break lines separating the segments
    text_labels : list of :class:`matplotlib.text.Text` objects
        List of added text labels

    """
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = []
    # Disable features that depend on a monotonic axis
    if monotonic_axis is None:
        labels, compact, add_breaks = [], False, False

    # Get segment startpoints and endpoints along monotonic axis
    if monotonic_axis == 'x':
        start = np.array([np.asarray(segm)[:, 0].min() for segm in segments])
        end = np.array([np.asarray(segm)[:, 0].max() for segm in segments])
    else:
        start = np.array([np.asarray(segm)[:, 1].min() for segm in segments])
        end = np.array([np.asarray(segm)[:, 1].max() for segm in segments])

    if compact:
        # Calculate offset between original and compacted coordinate, and adjust coordinates accordingly
        compacted_end = np.cumsum(end - start)
        compacted_start = np.array([0.0] + compacted_end[:-1].tolist())
        offset = start - compacted_start
        start, end = compacted_start, compacted_end
        # Redefine monotonic axis label formatter to add appropriate offset to label value, depending on segment
        class SegmentedScalarFormatter(mpl.ticker.ScalarFormatter):
            """Expand x axis value to correct segment before labelling."""
            def __init__(self, useOffset=True, useMathText=False):
                mpl.ticker.ScalarFormatter.__init__(self, useOffset, useMathText)
            def __call__(self, x, pos=None):
                segment = max(start.searchsorted(x, side='right') - 1, 0)
                return mpl.ticker.ScalarFormatter.__call__(self, x + offset[segment], pos)
        # Subtract segment offsets from appropriate coordinate
        if monotonic_axis == 'x':
            segments = [np.column_stack((np.asarray(segm)[:, 0] - offset[n], np.asarray(segm)[:, 1]))
                        for n, segm in enumerate(segments)]
            ax.xaxis.set_major_formatter(SegmentedScalarFormatter())
        else:
            segments = [np.column_stack((np.asarray(segm)[:, 0], np.asarray(segm)[:, 1] - offset[n]))
                        for n, segm in enumerate(segments)]
            ax.yaxis.set_major_formatter(SegmentedScalarFormatter())

    # Plot the segment lines as a collection
    segment_lines = mpl.collections.LineCollection(segments, **kwargs)
    ax.add_collection(segment_lines)

    segment_centers, breaks = (start + end) / 2, (start[1:] + end[:-1]) / 2
    break_lines, text_labels = None, []
    if monotonic_axis == 'x':
        # Break lines and labels have x coordinates fixed to data and y coordinates fixed to axes (like axvline)
        transFixedY = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for n, label in enumerate(labels):
            text_labels.append(ax.text(segment_centers[n], 0.02, label, transform=transFixedY,
                                       ha='center', va='bottom', clip_on=True))
        if add_breaks:
            break_lines = mpl.collections.LineCollection([[(s, 0), (s, 1)] for s in breaks], colors='k',
                                                         linewidths=0.5, linestyles='dotted', transform=transFixedY)
            ax.add_collection(break_lines)
        # Only set monotonic axis limits
        ax.set_xlim(start[0], end[-1])
        ax.autoscale_view(scalex=False)
    elif monotonic_axis == 'y':
        # Break lines and labels have x coordinates fixed to axes and y coordinates fixed to data (like axhline)
        transFixedX = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for n, label in enumerate(labels):
            text_labels.append(ax.text(0.02, segment_centers[n], label, transform=transFixedX,
                                       ha='left', va='center', clip_on=True))
        if add_breaks:
            break_lines = mpl.collections.LineCollection([[(0, s), (1, s)] for s in breaks], colors='k',
                                                         linewidths=0.5, linestyles='dotted', transform=transFixedX)
            ax.add_collection(break_lines)
        ax.set_ylim(start[0], end[-1])
        ax.autoscale_view(scaley=False)
    else:
        ax.autoscale_view()

    return segment_lines, break_lines, text_labels

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_segments
#--------------------------------------------------------------------------------------------------

def plot_segments(x, y, z=None, labels=None, width=0.0, compact=True, add_breaks=True,
                  monotonic_axis='auto', color=None, clim=None, ax=None, **kwargs):
    """Plot segmented lines, bars or images.

    This plots a sequence of lines, bars or images consisting of one or more
    *segments*. These segments are typically linearly arranged, either
    left-to-right along the *x*-axis or up-and-down along the *y*-axis,
    potentially with gaps between the segments.

    The axis along which the segments are arranged is called *monotonic* if the
    coordinates along that axis increase monotonically through the sequence of
    data segments. The classic example of a segmented plot is a sequence of line
    segments with the *y* coordinates representing time series data and the
    monotonic *x*-axis representing time. By default, the function attempts to
    detect the appropriate monotonic axis automatically.

    Because of their segmented nature, the plots share certain properties,
    regardless of whether lines, bars or images are drawn. Each segment may be
    labelled by a text string next to it. If *compact* is True, there will be no
    gaps between the segments along the monotonic axis. The tick labels on this
    axis are modified to reflect the original (padded) values. If *add_breaks*
    is True, the breaks between segments along the monotonic axis are indicated
    by dashed lines. If there is no monotonic axis, compaction and break lines
    are disabled and the text labels are placed near the start of each segment.

    Parameters
    ----------
    x, y : sequence of array-like, shape (N_k,) or (N_k, 2) or (M_k,)
        Coordinates of K segments along x and y axes. If the k'th segment has
        shape (N_k,) for both x and y axes, line segments will be drawn. If the
        k'th segment has shape (N_k,) for one axis and (N_k, 2) for the other
        axis, bar segments will be plotted with each bar aligned with the 2nd
        axis. If the k'th segment has shape (N_k,) for the x axis, (M_k,) for
        the y axis and (M_k, N_k) for the z axis, image segments will be plotted.
    z : sequence of array-like, shape (M_k, N_k), or None, optional
        Sequence of K 2-dimensional arrays to be plotted as image segments. If
        None, line or bar segments will be plotted.
    labels : sequence of strings, optional
        Corresponding sequence of text labels to add next to each segment along
        monotonic axis (default is no labels)
    width : float, optional
        If non-zero, replace contiguous line with staircase levels of specified
        width along monotonic axis
    compact : {True, False}, optional
        Plot with no gaps between segments along monotonic axis (only makes
        sense if there is such an axis)
    add_breaks : {True, False}, optional
        Add vertical (or horizontal) lines to indicate breaks between segments
        along monotonic axis (only makes sense if there is such an axis)
    monotonic_axis : {'auto', 'x', 'y', None}, optional
        Monotonic axis, along which segment coordinate increases monotonically
        (automatically detected by default)
    color : color spec or None, optional
        Color of segmented lines and bars
    clim : sequence of 2 floats, or None, optional
        Shared color limits of images, as (*vmin*, *vmax*). The default uses the
        the global minimum and maximum of all the arrays in *z*.
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)
    kwargs : dict, optional
        Extra keyword arguments are passed on to line collection constructor

    Returns
    -------
    segments : LineCollection, PolyCollection or list of AxesImage objects
        Segment plot object(s) of appropriate type, depending on whether lines,
        bars or images were plotted, respectively
    text_labels : list of :class:`matplotlib.text.Text` objects
        List of added text labels
    break_lines : :class:`matplotlib.collections.LineCollection` object, or None
        Collection of break lines separating the segments

    Raises
    ------
    ValueError
        If x, y and z coordinates have different lengths or mismatched shapes,
        so that they cannot be formed into lines, bars or images

    """
    if labels is None:
        labels = []
    # Attempt to detect monotonic axis if requested - look for axis with 1-dimensional data that is sorted
    if monotonic_axis == 'auto':
        monotonic = lambda x: np.all(np.array([np.ndim(s) for s in x]) == 1) and \
                              np.abs(np.sign(np.diff(np.hstack(x))).sum()) == np.sum([len(s) for s in x]) - 1
        monotonic_axis = 'x' if monotonic(x) else 'y' if monotonic(y) else None
    # Disable features that depend on a monotonic axis and multiple segments
    if monotonic_axis is None or ((len(x) == 1) and (len(y) == 1)):
        compact, add_breaks = False, False

    # Double-check coordinate shapes and select appropriate plot type
    if len(x) != len(y):
        raise ValueError('Different number of segments for x and y coordinates (%d vs %d)' % (len(x), len(y)))
    if z is None and [np.shape(s)[0] for s in x] != [np.shape(s)[0] for s in y]:
        raise ValueError('Shape mismatch between x and y (segment lengths are %s vs %s)' %
                         ([np.shape(s)[0] for s in x], [np.shape(s)[0] for s in y]))
    if z is not None:
        yx_shape = zip([np.shape(s)[0] for s in y], [np.shape(s)[0] for s in x])
        if [np.shape(s) for s in z] != yx_shape:
            raise ValueError('Shape mismatch between z and (y, x) (segment shapes are %s vs %s)' %
                             ([np.shape(s) for s in z], yx_shape))
        plot_type = 'image'
    elif np.isscalar(x[0][0]) and np.isscalar(y[0][0]):
        plot_type = 'line'
    elif np.isscalar(x[0][0]) and np.shape(y[0][0]) == (2,) and monotonic_axis != 'y':
        plot_type = 'barv'
    elif np.shape(x[0][0]) == (2,) and np.isscalar(y[0][0]) and monotonic_axis != 'x':
        plot_type = 'barh'
    else:
        raise ValueError('Could not figure out whether to plot lines, bars or images based on x, y, z shapes')

    # Only now get ready to plot...
    if ax is None:
        ax = plt.gca()

    # Get segment startpoints and endpoints along monotonic axis
    if monotonic_axis == 'x':
        if np.isscalar(width) and width == 0.0:
            width = np.array([np.mean(np.diff(segm)) for segm in x])
        start = np.array([np.min(segm) for segm in x]) - width / 2.0
        end = np.array([np.max(segm) for segm in x]) + width / 2.0
    elif monotonic_axis == 'y':
        if np.isscalar(width) and width == 0.0:
            width = np.array([np.mean(np.diff(segm)) for segm in y])
        start = np.array([np.min(segm) for segm in y]) - width / 2.0
        end = np.array([np.max(segm) for segm in y]) + width / 2.0

    # Handle compaction along monotonic axis
    if compact:
        # Calculate offset between original and compacted coordinate, and adjust coordinates accordingly
        compacted_end = np.cumsum(end - start)
        compacted_start = np.array([0.0] + compacted_end[:-1].tolist())
        offset = start - compacted_start
        start, end = compacted_start, compacted_end
        # Redefine monotonic axis label formatter to add appropriate offset to label value, depending on segment
        class SegmentedScalarFormatter(mpl.ticker.ScalarFormatter):
            """Expand x axis value to correct segment before labelling."""
            def __init__(self, useOffset=True, useMathText=False):
                mpl.ticker.ScalarFormatter.__init__(self, useOffset, useMathText)
            def __call__(self, x, pos=None):
                segment = max(start.searchsorted(x, side='right') - 1, 0)
                return mpl.ticker.ScalarFormatter.__call__(self, x + offset[segment], pos)
        # Subtract segment offsets from appropriate coordinate
        if monotonic_axis == 'x':
            x = [np.asarray(xsegm) - offset[n] for n, xsegm in enumerate(x)]
            ax.xaxis.set_major_formatter(SegmentedScalarFormatter())
        else:
            y = [np.asarray(ysegm) - offset[n] for n, ysegm in enumerate(y)]
            ax.yaxis.set_major_formatter(SegmentedScalarFormatter())

    # Plot the main line / bar / image segments
    if plot_type == 'line':
        if color is not None:
            kwargs['color'] = color
        segments = mpl.collections.LineCollection([zip(xsegm, ysegm) for xsegm, ysegm in zip(x, y)], **kwargs)
        ax.add_collection(segments)
    elif plot_type == 'barv':
        x = np.hstack(x)
        y1 = np.hstack([np.asarray(ysegm)[:, 0] for ysegm in y])
        y2 = np.hstack([np.asarray(ysegm)[:, 1] for ysegm in y])
        # Form makeshift rectangular patches (the factor 0.999 prevents glitches in bars)
        xxx = np.array([x - 0.999 * width / 2, x + 0.999 * width / 2, x]).transpose().ravel()
        yyy1, yyy2 = np.repeat(y1, 3), np.repeat(y2, 3)
        mask = np.arange(len(xxx)) % 3 == 2
        if color is not None:
            kwargs['facecolors'] = kwargs['edgecolors'] = color
        # Fill_between (which uses poly path) is much faster than a Rectangle patch collection
        segments = ax.fill_between(xxx, yyy1, yyy2, where=~mask, **kwargs)
    elif plot_type == 'barh':
        x1 = np.hstack([np.asarray(xsegm)[:, 0] for xsegm in x])
        x2 = np.hstack([np.asarray(xsegm)[:, 1] for xsegm in x])
        y = np.hstack(y)
        # Form makeshift rectangular patches
        yyy = np.array([y - 0.999 * width / 2, y + 0.999 * width / 2, y]).transpose().ravel()
        xxx1, xxx2 = np.repeat(x1, 3), np.repeat(x2, 3)
        mask = np.arange(len(yyy)) % 3 == 2
        if color is not None:
            kwargs['facecolors'] = kwargs['edgecolors'] = color
        segments = ax.fill_betweenx(yyy, xxx1, xxx2, where=~mask, **kwargs)
    elif plot_type == 'image':
        # The default color limits use the minimum and maximum values of z
        if clim is None:
            cmin, cmax = np.min([im.min() for im in z]), np.max([im.max() for im in z])
            crange = 1.0 if cmin == cmax else cmax - cmin
            clim = (cmin - 0.05 * crange, cmax + 0.05 * crange)
        segments = []
        colornorm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
        for xsegm, ysegm, zsegm in zip(x, y, z):
            norm_rgb = mpl.cm.jet(colornorm(zsegm))
            # if grey_rows is not None:
            #     norm_rgb_grey = mpl.cm.gray(colornorm(zsegm))
            #     norm_rgb[grey_rows, :, :] = norm_rgb_grey[grey_rows, :, :]
            # Calculate image extent, which assumes equispaced x and y ticks and defines the bounding box of image
            xdelta, ydelta = np.mean(np.diff(xsegm)), np.mean(np.diff(ysegm))
            extent = (xsegm[0] - xdelta / 2, xsegm[-1] + xdelta / 2, ysegm[0] - ydelta / 2, ysegm[-1] + ydelta / 2)
            # Convert image data to RGB uint8 form, to save space and speed things up
            # The downside is that color limits of image are fixed
            im = ax.imshow(np.uint8(np.round(norm_rgb * 255)), aspect='auto', origin='lower',
                           interpolation='nearest', extent=extent)
            # Save original normaliser to ensure correct colorbar() behaviour
            im.set_norm(colornorm)
            segments.append(im)

    ## MPL WORKAROUND ##
    # In matplotlib 1.0.0 and earlier (at least), adding break lines (axvline collection) messes up axes data limits
    # The y bottom limit is erroneously reset to 0 - as a workaround, do an initial autoscale_view
    ax.autoscale_view()
    # Add text labels and break lines, and set axes limits
    text_labels, break_lines = [], None
    # Break line styles differ for image and non-image plots
    break_kwargs = {'colors' : 'k', 'linewidths' : 2.0, 'linestyles' : 'solid'} if plot_type == 'image' else \
                   {'colors' : 'k', 'linewidths' : 0.5, 'linestyles' : 'dotted'}
    if monotonic_axis == 'x':
        # Break lines and labels have x coordinates fixed to data and y coordinates fixed to axes (like axvline)
        transFixedY = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for n, label in enumerate(labels):
            text_labels.append(ax.text((start[n] + end[n]) / 2, 0.02, label, transform=transFixedY,
                                       ha='center', va='bottom', clip_on=True, backgroundcolor='w'))
        if add_breaks:
            breaks = (start[1:] + end[:-1]) / 2
            break_lines = mpl.collections.LineCollection([[(s, 0), (s, 1)] for s in breaks],
                                                         transform=transFixedY, **break_kwargs)
            ax.add_collection(break_lines)
        # Only set monotonic axis limits - the other axis is autoscaled
        ax.set_xlim(start[0], end[-1])
        ax.autoscale_view(scalex=False)
    elif monotonic_axis == 'y':
        # Break lines and labels have x coordinates fixed to axes and y coordinates fixed to data (like axhline)
        transFixedX = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for n, label in enumerate(labels):
            text_labels.append(ax.text(0.02, (start[n] + end[n]) / 2, label, transform=transFixedX,
                                       ha='left', va='center', clip_on=True, backgroundcolor='w'))
        if add_breaks:
            breaks = (start[1:] + end[:-1]) / 2
            break_lines = mpl.collections.LineCollection([[(0, s), (1, s)] for s in breaks],
                                                         transform=transFixedX, **break_kwargs)
            ax.add_collection(break_lines)
        ax.set_ylim(start[0], end[-1])
        ax.autoscale_view(scaley=False)
    else:
        for n, label in enumerate(labels):
            # Add text label just before the start of segment, with white background to make it readable above data
            lx, ly = x[n][0] - 0.03 * (x[n][-1] - x[n][0]), y[n][0] - 0.03 * (y[n][-1] - y[n][0])
            text_labels.append(ax.text(lx, ly, label, ha='center', va='center', clip_on=True, backgroundcolor='w'))
        ax.autoscale_view()

    return segments, text_labels, break_lines

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_compacted_images
#--------------------------------------------------------------------------------------------------

def plot_compacted_images(imdata, xticks, labels=None, ylim=None, clim=None, grey_rows=None, ax=None):
    """Plot sequence of images in compacted form.

    This plots a sequence of 2-D arrays (with the same number of rows but
    possibly varying number of columns) as images on a single set of axes, with
    no gaps between the images along the *x* axis. Each image has an associated
    sequence of *x* ticks, with one tick per image column. The tick labels on the
    *x* axis are modified to reflect the original (padded) tick values, and the
    breaks between segments are indicated by vertical lines. Some of the image
    rows may optionally be greyed out (e.g. to indicate RFI-corrupted channels).

    Parameters
    ----------
    imdata : sequence of array-like, shape (M, N_k)
        Sequence of 2-D arrays (*image0*, *image1*, ..., *imagek*, ..., *imageK*)
        to be displayed as images, where each array has the same number of rows,
        *M*, but a potentially unique number of columns, *N_k*
    xticks : sequence of array-like, shape (N_k,)
        Sequence of 1-D arrays (*x_0*, *x_1*, ..., *x_k*, ..., *x_K*) serving as
        *x*-axis ticks for the corresponding images, where *x_k* has length *N_k*
    labels : sequence of strings, optional
        Corresponding sequence of text labels to add below each image
    ylim : sequence of 2 floats, or None, optional
        Shared *y* limit of images, as (*ymin*, *ymax*), based on their common
        rows (default is (1, M))
    clim : sequence of 2 floats, or None, optional
        Shared color limits of images, as (*vmin*, *vmax*). The default uses the
        the global minimum and maximum of all the arrays in *imdata*.
    grey_rows : sequence of integers, optional
        Sequence of indices of rows which will be greyed out in each image
        (default is no greyed-out rows)
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)

    Returns
    -------
    images : list of :class:`matplotlib.image.AxesImage` objects
        List of images
    border_lines : :class:`matplotlib.collections.LineCollection` object
        Collection of vertical lines separating the images
    text_labels : list of :class:`matplotlib.text.Text` objects
        List of added text labels

    """
    if ax is None:
        ax = plt.gca()
    if clim is None:
        cmin = np.min([im.min() for im in imdata])
        cmax = np.max([im.max() for im in imdata])
        crange = cmax - cmin
        if crange == 0.0:
            crange = 1.0
        clim = (cmin - 0.05 * crange, cmax + 0.05 * crange)
    if ylim is None:
        ylim = (1, imdata[0].shape[0])
    if labels is None:
        labels = []
    start = np.array([x.min() for x in xticks])
    end = np.array([x.max() for x in xticks])
    compacted_start = [0.0] + np.cumsum(end - start).tolist()
    x_origin = start.min()
    images = []
    for k, (x, im) in enumerate(zip(xticks, imdata)):
        colornorm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
        image_data = mpl.cm.jet(colornorm(im))
        if grey_rows is not None:
            image_data_grey = mpl.cm.gray(colornorm(im))
            image_data[grey_rows, :, :] = image_data_grey[grey_rows, :, :]
        images.append(ax.imshow(np.uint8(np.round(image_data * 255)), aspect='auto',
                                interpolation='nearest', origin='lower',
                                extent=(x[0] - start[k] + compacted_start[k],
                                        x[-1] - start[k] + compacted_start[k], ylim[0], ylim[1])))
    # These border lines have x coordinates fixed to the data and y coordinates fixed to the axes
    transFixedY = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    border_lines = mpl.collections.LineCollection([[(s, 0), (s, 1)] for s in compacted_start[1:-1]],
                                                  colors='k', linewidths=2.0, linestyles='solid',
                                                  transform=transFixedY)
    ax.add_collection(border_lines)
    ax.axis([compacted_start[0], compacted_start[-1], ylim[0], ylim[1]])
    text_labels = []
    for k, label in enumerate(labels):
        text_labels.append(ax.text(np.mean(compacted_start[k:k+2]), 0.02, label, transform=transFixedY,
                                   ha='center', va='bottom', clip_on=True, color='w'))
    # Redefine x-axis label formatter to display the correct time for each segment
    class SegmentedScalarFormatter(mpl.ticker.ScalarFormatter):
        """Expand x axis value to correct segment before labelling."""
        def __init__(self, useOffset=True, useMathText=False):
            mpl.ticker.ScalarFormatter.__init__(self, useOffset, useMathText)
        def __call__(self, x, pos=None):
            if x > compacted_start[0]:
                segment = (compacted_start[:-1] < x).nonzero()[0][-1]
                x = x - compacted_start[segment] + start[segment] - x_origin
            return mpl.ticker.ScalarFormatter.__call__(self, x, pos)
    ax.xaxis.set_major_formatter(SegmentedScalarFormatter())
    return images, border_lines, text_labels

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_marker_3d
#---------------------------------------------------------------------------------------------------------

def plot_marker_3d(x, y, z, max_size=0.75, min_size=0.05, marker_type='scatter', num_lines=8, ax=None, **kwargs):
    """Pseudo-3D scatter plot using marker size to indicate height.

    This plots markers at given ``(x, y)`` positions, with marker size determined
    by *z* values. This is an alternative to :func:`matplotlib.pyplot.pcolor`,
    with the advantage that the *x* and *y* values do not need to be on a regular
    grid, and that it is easier to compare the relative size of *z* values. The
    disadvantage is that the markers may have excessive overlap or very small
    sizes, which obscures the plot. This can be controlled by the max_size and
    min_size parameters.

    Parameters
    ----------
    x : sequence
        Sequence of *x* coordinates of markers
    y : sequence
        Sequence of *y* coordinates of markers
    z : sequence
        Sequence of *z* heights, transformed to marker size
    max_size : float, optional
        Radius of biggest marker, relative to average spacing between markers
    min_size : float, optional
        Radius of smallest marker, relative to average spacing between markers
    marker_type : {'scatter', 'circle', 'asterisk'}, optional
        Type of marker
    num_lines : int, optional
        Number of lines in asterisk
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)
    kwargs : dict, optional
        Extra keyword arguments are passed on to underlying plot function

    Returns
    -------
    handle : handle or list
        Handle of asterisk line, list of circle patches, or scatter collection

    Raises
    ------
    ValueError
        If marker type is unknown

    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    assert max_size >= min_size, "In plot_marker_3d, min_size should not be bigger than max_size."
    if ax is None:
        ax = plt.gca()

    # Normalise z to lie between 0 and 1
    z = (z - z.min()) / (z.max() - z.min())
    # Threshold z, so that the minimum size will have the desired ratio to the maximum size
    z[z < min_size/max_size] = min_size/max_size
    # Determine median spacing between vectors
    min_dist = np.zeros(len(x))
    for ind in xrange(len(x)):
        dist_sq = (x - x[ind]) ** 2 + (y - y[ind]) ** 2
        min_dist[ind] = np.sqrt(dist_sq[dist_sq > 0].min())
    # Scale z so that maximum value is desired factor of median spacing
    z *= max_size * np.median(min_dist)

    if marker_type == 'asterisk':
        # Use random initial angles so that asterisks don't overlap in regular pattern, which obscures their size
        ang = np.pi * np.random.random_sample(z.shape)
        x_asterisks, y_asterisks = [], []
        # pylint: disable-msg=W0612
        for side in range(num_lines):
            x_dash = np.vstack((x - z * np.cos(ang), x + z * np.cos(ang), np.tile(np.nan, x.shape))).transpose()
            y_dash = np.vstack((y - z * np.sin(ang), y + z * np.sin(ang), np.tile(np.nan, y.shape))).transpose()
            x_asterisks += x_dash.ravel().tolist()
            y_asterisks += y_dash.ravel().tolist()
            ang += np.pi / num_lines
        # All asterisks form part of one big line...
        return ax.plot(x_asterisks, y_asterisks, **kwargs)

    elif marker_type == 'circle':
        # Add a circle patch for each marker
        for ind in xrange(len(x)):
            ax.add_patch(mpl.patches.Circle((x[ind], y[ind]), z[ind], **kwargs))
        return ax.patches

    elif marker_type == 'scatter':
        # Get axes size in points
        points_per_axis = ax.get_position().extents[2:] * ax.get_figure().get_size_inches() * 72.0
        # Get points per data units in x and y directions
        x_range, y_range = 1.1 * (x.max() - x.min()), 1.1 * (y.max() - y.min())
        points_per_data = points_per_axis / np.array((x_range, y_range))
        # Scale according to largest data axis
        z *= points_per_data.min()
        return ax.scatter(x, y, 20.0 * z ** 2, **kwargs)

    else:
        raise ValueError("Unknown marker type '" + marker_type + "'")

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  gaussian_ellipses
#---------------------------------------------------------------------------------------------------------

def gaussian_ellipses(mean, cov, contour=0.5, num_points=200):
    """Contour ellipses of two-dimensional Gaussian function.

    Parameters
    ----------
    mean : real array-like, shape (2,)
        Two-dimensional mean vector
    cov : real array-like, shape (2, 2)
        Two-by-two covariance matrix
    contour : float, or real array-like, shape (*K*,), optional
        Contour height of ellipse(s), as a (list of) factor(s) of the peak value.
        For a factor *sigma* of standard deviation, use ``exp(-0.5 * sigma**2)``.
    num_points : int, optional
        Number of points *N* on each ellipse

    Returns
    -------
    ellipses : real array, shape (*K*, *N*, 2)
        Array containing 2-D ellipse coordinates

    Raises
    ------
    ValueError
        If mean and/or cov has wrong shape

    """
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    contour = np.atleast_1d(np.asarray(contour))
    if (mean.shape != (2,)) or (cov.shape != (2, 2)):
        raise ValueError('Mean and covariance should be 2-dimensional, with shapes (2,) and (2, 2) instead of'
                         + str(mean.shape) + ' and ' + str(cov.shape))
    # Create parametric circle
    t = np.linspace(0.0, 2.0 * np.pi, num_points)
    circle = np.vstack((np.cos(t), np.sin(t)))
    # Determine and apply transformation to ellipse
    eig_val, eig_vec = np.linalg.eig(cov)
    circle_to_ellipse = np.dot(eig_vec, np.diag(np.sqrt(eig_val)))
    base_ellipse = np.real(np.dot(circle_to_ellipse, circle))
    ellipses = []
    for cnt in contour:
        ellipse = np.sqrt(-2.0 * np.log(cnt)) * base_ellipse + mean[:, np.newaxis]
        ellipses.append(ellipse.transpose())
    return np.array(ellipses)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_db_contours
#--------------------------------------------------------------------------------------------------

def plot_db_contours(x, y, Z, levels=None, sin_coords=False, add_lines=True, ax=None):
    """Filled contour plot of 2-D spherical function in decibels.

    The spherical function ``z = f(x, y)`` is a function of two angles, *x* and
    *y*, given in degrees. The function should be real-valued, but may contain
    negative parts. These are indicated by dashed contours. The contour levels
    are based on the absolute value of *z* in dBs.

    Parameters
    ----------
    x : real array-like, shape (N,)
        Vector of x coordinates, in degrees
    y : real array-like, shape (M,)
        Vector of y coordinates, in degrees
    Z : real array-like, shape (M, N)
        Matrix of z values, with rows associated with *y* and columns with *x*
    levels : real array-like, shape (K,), optional
        Sequence of ascending contour levels, in dB (default ranges from -60 to 0)
    sin_coords : {False, True}, optional
        True if coordinates should be converted to projected sine values. This
        is useful if a large portion of the sphere is plotted.
    add_lines : {True, False}, optional
        True if contour lines should be added to plot
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)

    Returns
    -------
    cset : :class:`matplotlib.contour.ContourSet` object
        Set of filled contour regions (useful for setting up color bar)

    """
    # pylint: disable-msg=C0103
    if ax is None:
        ax = plt.gca()
    if levels is None:
        levels = np.linspace(-60.0, 0.0, 21)
    levels = np.sort(levels)
    # Crude corner cutouts to indicate region outside spherical projection
    quadrant = np.linspace(0.0, np.pi / 2.0, 401)
    corner_x = np.concatenate([np.cos(quadrant), [1.0, 1.0]])
    corner_y = np.concatenate([np.sin(quadrant), [1.0, 0.0]])
    if sin_coords:
        x, y = np.sin(x * np.pi / 180.0), np.sin(y * np.pi / 180.0)
    else:
        corner_x, corner_y = 90.0 * corner_x, 90.0 * corner_y
    Z_db = 10.0 * np.log10(np.abs(Z))
    # Remove -infs (keep above lowest contour level to prevent white patches in contourf)
    Z_db[Z_db < levels.min() + 0.01] = levels.min() + 0.01
    # Also keep below highest contour level for the same reason
    Z_db[Z_db > levels.max() - 0.01] = levels.max() - 0.01

    cset = ax.contourf(x, y, Z_db, levels)
    mpl.rc('contour', negative_linestyle='solid')
    if add_lines:
        # Non-negative function has straightforward contours
        if Z.min() >= 0.0:
            ax.contour(x, y, Z_db, levels, colors='k', linewidths=0.5)
        else:
            # Indicate positive parts with solid contours
            Z_db_pos = Z_db.copy()
            Z_db_pos[Z < 0.0] = levels.min() + 0.01
            ax.contour(x, y, Z_db_pos, levels, colors='k', linewidths=0.5)
            # Indicate negative parts with dashed contours
            Z_db_neg = Z_db.copy()
            Z_db_neg[Z > 0.0] = levels.min() + 0.01
            mpl.rc('contour', negative_linestyle='dashed')
            ax.contour(x, y, Z_db_neg, levels, colors='k', linewidths=0.5)
    if sin_coords:
        ax.set_xlabel(r'sin $\theta$ sin $\phi$')
        ax.set_ylabel(r'sin $\theta$ cos $\phi$')
    else:
        ax.set_xlabel('x (deg)')
        ax.set_ylabel('y (deg)')
    ax.axis('image')
    ax.fill( corner_x,  corner_y, facecolor='w')
    ax.fill(-corner_x,  corner_y, facecolor='w')
    ax.fill(-corner_x, -corner_y, facecolor='w')
    ax.fill( corner_x, -corner_y, facecolor='w')
    return cset

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  save_fits_image
#--------------------------------------------------------------------------------------------------

def save_fits_image(filename, x, y, Z, target_name='', coord_system='radec',
                    projection_type='ARC', data_unit='', freq_Hz=0.,
                    bandwidth_Hz=0., pol=None, observe_date='', create_date='',
                    telescope='', instrument='', observer='', clobber=False):
    """Save image data to FITS file.

    This produces a FITS file from 2D spatial image data. It optionally allows
    degenerate frequency and Stokes axes, in order to specify the RF frequency
    and polarisation of the image data.

    Parameters
    ----------
    filename : string
        Name of FITS file to create
    x : real array-like, shape (N,)
        Vector of x coordinates, in degrees
    y : real array-like, shape (M,)
        Vector of y coordinates, in degrees
    Z : real array-like, shape (M, N)
        Matrix of z values, with rows associated with *y* and columns with *x*
    target_name : string, optional
        Name of source being imaged
    coord_system : {'radec', 'azel', 'lm'}, optional
        Spherical coordinate system serving as basis for *x* and *y*
    projection_type : {'ARC', 'SIN', 'TAN', 'STG', 'CAR'}, optional
        Type of spherical projection used to obtain *x*-*y* plane
    data_unit : string, optional
        Unit of z data
    freq_Hz : float, optional
        Centre frequency of z data, in Hz (if bigger than 0, add a frequency axis)
    bandwidth_Hz : float, optional
        Bandwidth of z data, in Hz
    pol : {None, 'I', 'Q', 'U', 'V', 'HH', 'VV', 'VH', 'HV', 'XX', 'YY', 'XY', 'YX'}, optional
        Polarisation of z data (if not None, add a Stokes axis)
    observe_date : string, optional
        UT timestamp of start of observation (format YYYY-MM-DD[Thh:mm:ss[.sss]])
    create_date : string, optional
        UT timestamp of file creation (format YYYY-MM-DD[Thh:mm:ss[.sss]])
    telescope : string, optional
        Telescope that performed the observation
    instrument : string, optional
        Instrument that recorded the data
    observer : string, optional
        Person responsible for the observation
    clobber : {False, True}, optional
        True if FITS file should be replaced if it already exists

    Raises
    ------
    ValueError
        If coordinate system is unknown

    """
    if coord_system == 'azel':
        axes = ['AZ---' + projection_type, 'EL---' + projection_type]
    elif coord_system == 'radec':
        axes = ['RA---' + projection_type, 'DEC--' + projection_type]
    elif coord_system == 'lm':
        axes = ['L', 'M']
    else:
        raise ValueError('Unknown coordinate system for FITS image')
    if data_unit == 'counts':
        data_unit = 'count'
    # Pick centre pixel as reference, out of convenience
    ref_pixel = [(len(x) // 2 + 1), (len(y) // 2) + 1]
    ref_world = [x[ref_pixel[0] - 1], y[ref_pixel[1] - 1]]
    world_per_pixel = [(x[-1] - x[0]) / (len(x) - 1), (y[-1] - y[0]) / (len(y) - 1)]
    # If frequency is specified, add a frequency axis
    if freq_Hz > 0:
        axes.append('FREQ')
        ref_pixel.append(1)
        ref_world.append(freq_Hz)
        world_per_pixel.append(bandwidth_Hz)
        Z = Z[np.newaxis]
    # If polarisation is specified, add a Stokes axis
    if pol is not None:
        stokes_code = {'I' : 1, 'Q' : 2, 'U' : 3, 'V' : 4,
                       'XX' : -5, 'YY' : -6, 'XY' : -7, 'YX' : -8,
                       'HH' : -5, 'VV' : -6, 'HV' : -7, 'VH' : -8}
        axes.append('STOKES')
        ref_pixel.append(1)
        ref_world.append(stokes_code[pol])
        world_per_pixel.append(1)
        Z = Z[np.newaxis]

    phdu = pyfits.PrimaryHDU(Z)
    phdu.update_header()
    phdu.header.update('DATE', create_date, comment='UT file creation time')
    phdu.header.update('ORIGIN', 'SKA SA', 'institution that created file')
    phdu.header.update('DATE-OBS', observe_date, comment='UT observation start time')
    phdu.header.update('TELESCOP', telescope)
    phdu.header.update('INSTRUME', instrument)
    phdu.header.update('OBSERVER', observer)
    phdu.header.update('OBJECT', target_name, comment='source name')
    phdu.header.update('EQUINOX', 2000.0, comment='equinox of ra dec')
    phdu.header.update('BUNIT', data_unit, comment='units of flux')
    for n, (ax_type, ref_pix, ref_val, ref_delt) in enumerate(zip(axes, ref_pixel, ref_world, world_per_pixel)):
        phdu.header.update('CTYPE%d' % (n+1), ax_type)
        phdu.header.update('CRPIX%d' % (n+1), ref_pix)
        phdu.header.update('CRVAL%d' % (n+1), ref_val)
        phdu.header.update('CDELT%d' % (n+1), ref_delt)
        phdu.header.update('CROTA%d' % (n+1), 0)
    phdu.header.update('DATAMAX', Z.max(), comment='max pixel value')
    phdu.header.update('DATAMIN', Z.min(), comment='min pixel value')
    pyfits.writeto(filename, phdu.data, phdu.header, clobber=clobber)
