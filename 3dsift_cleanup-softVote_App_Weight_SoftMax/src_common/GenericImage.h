
//
// File: GenericImage.h
// Developer: Matt Toews
// Desc: A generic image class.
//

#ifndef __GENERICIMAGE_H__
#define __GENERICIMAGE_H__

class GenericImage
{
public:

	GenericImage();

	~GenericImage();

	//
	// Cols()
	//
	// Returns the number of columns in the image.
	//
	int
		Cols(
		) const;

	//
	// Rows()
	//
	// Returns the number of rows in the image.
	//
	int
		Rows(
		) const;

	//
	// RowInc()
	//
	// Returns the row increment for the iamge.
	//
	int
		RowInc(
		) const;

	//
	// BitsPerPixel()
	//
	// Returns the bits per pixel in the image.
	//
	int
		BitsPerPixel(
		) const;

	//
	// ImageRow()
	//
	// Returns a pointer to the data at image row iRow.
	//
	unsigned char *
		ImageRow(
		int iRow
		) const;

	//
	// Initialize a generic image, allocates memory
	//
	int
		Initialize(
		const int iRows,
		const int iCols,
		const int iRowInc,
		const int iBitsPerPixel
		);

	//
	// Initialize a generic image, allocates memory,
	// same dimensions as giExample
	//
	int
		Initialize(
		const GenericImage &giExample,
		int	bCopyBits = 0
		);

	//
	// Initialized a reference to a sub image (region within an image)
	//
	int
		InitializeSubImage(
		const int iRows,
		const int iCols,
		const int iRowInc,
		const int iBitsPerPixel,
		const unsigned char *pData
		);

	//
	// WriteToFile()
	//
	// Writes this image to a file (portable pixel format)
	//
	int
	WriteToFile(
		const char *pcFileName
		) const;

	//
	// Writes this image to a file (portable pixel format)
	//
	int
	WriteToFile(
	  void *_outFILE,
	  int iBinary = 1
	) const;

	//
	// Reallocates image data size.
	//

	int
		Reallocate(
		const int iBytes
		);

	//
	// MirrorImage()
	// Mirrors an image across the vertical midpoint.
	// (for calculating relection-invariance)
	//
	int
	MirrorImage(
	);

	//
	// MirrorImage()
	// Mirrors an image across the horizontal midpoint.
	// (for Windows display)
	//
	int
	MirrorImageVertical(
	);

	//
	// ExchangeRowsCols()
	//
	// Exchange image dimensions.
	//
	int
	ExchangeRowsCols(
	);

	//
	// SubSample()
	//
	// Sub sample this image into imgOut, reduce by 2X
	//
	int
	SubSample(
		GenericImage &imgOut
	);

	//
	// ReverseColors()
	//
	// Reverses the red and the blue colors (for Windows display)
	// of the image.
	//
	int
	ReverseColors(
		);

protected:

	int m_iCols;			// Columns/width of image (pixels)
	int	m_iRows;			// Rows/height of image	(pixels)
	int	m_iRowInc;			// Inc value for 1 row of data - bytes
	int	m_iBitsPerPixel;	// Bits per pixel

	unsigned char *m_pData;
	int m_iDataSizeBytes;
};

#ifndef _DEBUG

inline int
GenericImage::Cols() const
{
	return m_iCols;
}

inline int
GenericImage::Rows() const
{
	return m_iRows;
}

inline int
GenericImage::RowInc() const
{
	return m_iRowInc;
}

inline int
GenericImage::BitsPerPixel() const
{
	return m_iBitsPerPixel;
}

inline unsigned char *
GenericImage::ImageRow( int iRow ) const
{
	return(m_pData + iRow*m_iRowInc);
}

#endif

#endif
