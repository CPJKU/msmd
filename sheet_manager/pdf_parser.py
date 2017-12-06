"""This module implements a class that..."""
from __future__ import print_function

import logging
import re
import sys

import numpy

# from pdfminer.psparser import PSKeyword, PSLiteral, LIT
from pdfminer.pdfparser import PDFParser  # , PDFStreamParser
from pdfminer.pdfdocument import PDFDocument  # , PDFNoOutlines
from pdfminer.pdftypes import PDFObjectNotFound  # , PDFValueError
# from pdfminer.pdftypes import PDFStream, PDFObjRef, resolve1, stream_value
from pdfminer.pdfpage import PDFPage
# from pdfminer.utils import isnumber

from muscima.cropobject import CropObject

ESC_PAT = re.compile(r'[\000-\037&<>()"\042\047\134\177-\377]')
def e(s):
    return ESC_PAT.sub(lambda m:'&#%d;' % ord(m.group(0)), s)

__version__ = "0.0.1"


def parse_pdf(fname, target_width=None, with_links=False,
              collection_name=None, score_name=None):
    """Extracts the notehead centroid coordinates from the given
    LilyPond-generated PDF file, notehead bounding box coordinates,
    and builds the corresponding CropObjects.

    :param fname: Name of the input PDF file. Assumes it was generated
        from LilyPond with the option

        ``-e"(set-option 'point-and-click '(note-event))"``.

    :param target_width: The width of an image against which
        we want the centroids to be valid. Based on the PDF page
        size(s), the function will compute a ratio by which to scale
        the centroid coordinates from the PDF page, so that after resizing
        the page image to the given target width (without changing its
        aspect ratio), the centroids will corresponds to the object
        positions in this resized image. If not given, assumes no resizing
        will take place. (Can deal with a PDF where the pages have different
        sizes.)

    :param with_links: Also return links to the corresponding places
        in the originating LilyPond file. [NOT IMPLEMENTED]

    :returns: A triplet of per-page lists: centroids, bounding boxes,
        and CropObjects (MuNG data format for OMR; see the ``muscima``
        package).

        The returned objects are dictionaries per page.
        The dict keys are page numbers (``int``) starting from 0,
        the values are numpy arrays of the shape ``(n_notes, 2)``
        where the first coordinate is the **row**,
        the second is the **column**. The centroid dict values are
        ``[row, column]`` coordinates; the bounding box values are
        ``[top, left, bottom, right]`` lists, and the cropobjects
        are a list of ``CropObject`` instances (all initialized
        with ``clsname="notehead-full"``).

        If ``with_links`` is set, the CropObject ``data`` attribute
        has a ``lilypond_link`` to the location of the corresponding
        note's encoding in the lilypond file.

        Note that the CropObjects' ``objid`` attribute is set
        so that they do not collide across pages. However, CropObjects
        from various pages have the page number added to their
        ``document_name`` component of their UID. Keep this in mind
        if you want to merge them later.

    """
    centroids = dict()
    bboxes = dict()

    cropobjects_per_page = dict()
    _current_objid = 0  # We keep the OBJID
    if collection_name is None:
        collection_name = CropObject.UID_DEFAULT_DATASET_NAMESPACE
    if score_name is None:
        score_name = CropObject.UID_DEFAULT_DOCUMENT_NAMESPACE

    page_no = -1
    fp = file(fname, 'rb')

    # ??? What was this doing here?
    # pages = PDFPage.get_pages(fp)
    # for page in pages:
    #     parser = PDFStreamParser(page.contents[0].data)
    #     break
    parser = PDFParser(fp)
    doc = PDFDocument(parser)

    pages = [p for p in PDFPage.get_pages(fp)]
    page_height, page_width = -1, -1
    target_height = None
    scale = 1.0

    visited = set()

    for xref in doc.xrefs:
        for objid in xref.get_objids():

            if objid in visited: continue
            visited.add(objid)

            try:
                obj = doc.getobj(objid)
                if obj is None:
                    continue

                if not obj.__class__ is dict:
                    continue

                # Next page
                if 'Annots' in obj:
                    page_no += 1
                    bboxes[page_no] = []
                    cropobjects_per_page[page_no] = []

                    page = pages[page_no]
                    page_height = int(numpy.round(page.mediabox[3]))
                    page_width = int(numpy.round(page.mediabox[2]))

                    if target_width is not None:
                        scale = float(target_width) / page_width
                        target_height = page_height * scale
                    else:
                        target_height = page_height

                if not obj.has_key('Rect'):
                    continue

                bb_coords = obj['Rect']

                # Rescaling to target size
                if target_width is not None:
                    bb_coords = [c * scale for c in bb_coords]

                link_txt = obj['A']['URI']

                # Not a link to a note event!
                if link_txt.count(':') != 4:
                    continue

                # Initializing the CropObject.
                t, l, b, r = target_height - bb_coords[3], bb_coords[0], \
                             target_height - bb_coords[1], bb_coords[2]
                # print('Bounding box: {0} from coords {1}'.format((t, l, b, r), bb_coords))
                t_i, l_i, b_i, r_i = CropObject.bbox_to_integer_bounds(t, l, b, r)
                h_i, w_i = b_i - t_i, r_i - l_i
                mask = numpy.ones((h_i, w_i), dtype='uint8')

                uid = CropObject.build_uid(collection_name,
                                           score_name + '-P{0:02d}'.format(page_no),
                                           _current_objid)
                logging.debug('Creating CropObject with uid {0}'.format(uid))
                data = {'ly_link': link_txt}
                cropobject = CropObject(objid=_current_objid,
                                        clsname='notehead-full',
                                        top=t_i, left=l_i, height=h_i, width=w_i,
                                        mask=mask,
                                        uid=uid,
                                        data=data)
                cropobjects_per_page[page_no].append(cropobject)
                _current_objid += 1

                bboxes[page_no].append((t, l, b, r))

                # Recompute bboxes to centroids
                page_bboxes = numpy.asarray(bboxes[page_no])
                # For centroids, use the original float coordinates
                page_centroids = numpy.array([
                                 [(b + t) / 2. - 0.5,
                                  (l + r) / 2. - 0.5]
                                 for t, l, b, r in page_bboxes])
                centroids[page_no] = page_centroids

            except PDFObjectNotFound as e:
                sys.stderr.write('PDF object not found: %r' % e)

    return centroids, bboxes, cropobjects_per_page


def pdf2coords(fname, target_width=None, with_links=False):
    centroids, bboxes, cropobjects_per_page = parse_pdf(fname,
                                                        target_width=target_width,
                                                        with_links=with_links)
    return centroids