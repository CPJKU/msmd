"""This module implements a class that..."""
from __future__ import print_function

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


ESC_PAT = re.compile(r'[\000-\037&<>()"\042\047\134\177-\377]')
def e(s):
    return ESC_PAT.sub(lambda m:'&#%d;' % ord(m.group(0)), s)

__version__ = "0.0.1"


def pdf2coords(fname, target_width=None, with_links=False):
    """Extracts the notehead centroid coordinates from the given
    LilyPond-generated PDF file.

    :param fname: Name of the input PDF file. Assumes it was generated
        from LilyPond with the option ``-e"(set-option 'point-and-click '(note-event))"``.

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

    :returns: A dictionary of notehead centroid coordinates per page.
        The dict keys are page numbers (``int``) starting from 0,
        the values are numpy arrays of the shape ``(n_notes, 2)``
        where the first coordinate is the **row**,
        the second is the **column**.

        If ``with_links`` is set, also returns a dict of per-centroid links
        to the lilypond file.
    """
    centroids = dict()
    bboxes = dict()

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

                bboxes[page_no].append(bb_coords)

                # Recompute bboxes to centroids
                page_bboxes = numpy.asarray(bboxes[page_no])
                page_centroids = numpy.array([
                                 [target_height - (bb[1] + bb[3]) / 2. - 0.5,
                                  (bb[0] + bb[2]) / 2. - 0.5]
                    for bb in page_bboxes])
                centroids[page_no] = page_centroids

            except PDFObjectNotFound as e:
                sys.stderr.write('PDF object not found: %r' % e)

    return centroids
