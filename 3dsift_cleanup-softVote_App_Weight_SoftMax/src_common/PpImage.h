
//
// File: Histogram.h
// Developer: Matt Toews
// Desc: Defines an image class related to the portable pixel formats
//		(PPM, PGM).
//

#ifndef __PPIMAGE_H__
#define __PPIMAGE_H__

#include "GenericImage.h"

class PpImage : public GenericImage
{
public:

	PpImage();

	~PpImage();

	//
	// InitFromStream()
	//
	// Reads an image from a PPM file.
	//
	int
	InitFromStream( const char *pcFileName );

private:

	// Max pixel value

	int		m_iMaxPixelValue;

	// Portable pixel format image code: 'P*'

	char	m_cCode;

};

#endif
