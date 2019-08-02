
#include <assert.h>
#include <stdio.h>
#include <memory.h>

#include "GenericImage.h"

#define FALSE 0
#define TRUE 1

GenericImage::GenericImage()
{
	m_iCols = 0;			// Columns/width of image (pixels)
	m_iRows = 0;			// Rows/height of image	(pixels)
	m_iRowInc = 0;			// Inc value for 1 row of data - bytes
	m_iBitsPerPixel = 0;	// Bits per pixel
	m_pData = 0;
	m_iDataSizeBytes = 0;
}

GenericImage::~GenericImage()
{
	if( m_pData != 0 )
	{
		if( m_iDataSizeBytes != 0 )
		{
			delete [] m_pData;
		}
	}
}

int
GenericImage::Reallocate(
						 const int iBytes
						 )
{
	if( iBytes <= m_iDataSizeBytes )
	{
		// Array big enough
		return TRUE;
	}

	if( m_pData )
	{
		delete [] m_pData;
		m_pData = 0;
		m_iDataSizeBytes = 0;
	}

	m_pData = new unsigned char[ iBytes ];
	if( !m_pData )
	{
		// Error creating new array
		return FALSE;
	}

	m_iDataSizeBytes = iBytes;

	// New array created

	return TRUE;
}

int
GenericImage::Initialize(
	const int iRows,
	const int iCols,
	const int iRowInc,
	const int iBitsPerPixel
	)
{
	m_iCols = iCols;
	m_iRows = iRows;
	m_iRowInc = iRowInc;
	m_iBitsPerPixel = iBitsPerPixel;

	if( !Reallocate( m_iRows*m_iRowInc ) )
	{
		return FALSE;
	}

	return TRUE;
}

int
GenericImage::Initialize(
						 const GenericImage &giExample,
						 int bCopyBits
						 )
{
	int iReturn = Initialize(
		giExample.Rows(), giExample.Cols(),
		giExample.RowInc(), giExample.BitsPerPixel()
		);

	if( iReturn && bCopyBits )
	{
		memcpy( m_pData, giExample.ImageRow(0), giExample.Rows()*giExample.RowInc() );
	}

	return iReturn;
}

int
GenericImage::InitializeSubImage(
	const int iRows,
	const int iCols,
	const int iRowInc,
	const int iBitsPerPixel,
	const unsigned char *pData
	)
{
	m_iCols = iCols;
	m_iRows = iRows;
	m_iRowInc = iRowInc;
	m_iBitsPerPixel = iBitsPerPixel;

	assert( pData );

	// Delete allocated data

	if( m_iDataSizeBytes != 0 )
	{
		assert( m_pData != 0 );
		delete [] m_pData;
		m_iDataSizeBytes = 0;
	}

	m_pData = (unsigned char *)pData;

	return TRUE;
}

int
GenericImage::WriteToFile(
	  void *_outFILE,
	  int iBinary
	) const
{
	FILE *outfile = (FILE*)_outFILE;
	if( iBinary )
	{
		if( BitsPerPixel() == 32 )
		{
			// Float
			fprintf( outfile, "P8\n%d %d\n%d\n", m_iCols, m_iRows, 0 );
			for( int i = 0; i < m_iRows; i++ )
			{
				fwrite( ImageRow( i ), 1, m_iCols*sizeof(float), outfile );
			}

		}
		if( BitsPerPixel() == 24 )
		{
			// For some screwy reason, RGB must be BGR for PPM format

			fprintf( outfile, "P6\n%d %d\n%d\n", m_iCols, m_iRows, 255 );

			for( int i = 0; i < m_iRows; i++ )
			{
				fwrite( ImageRow( i ), 1, m_iCols*3, outfile );
			}
		}
		else if( BitsPerPixel() == 16 )
		{
			// For now, lets not allow this...
			//assert( FALSE );
			//fclose( outfile );
			//return FALSE;

			//int iMaxPixel, iMinPixel;
			//int iDivisor = iMaxPixel - iMinPixel + 1;

			// Treat like a greyscale image, although this could be color
			fprintf( outfile, "P5\n%d %d\n%d\n", m_iCols, m_iRows, 255 );

			for( int i = 0; i < m_iRows; i++ )
			{
				unsigned short *psRow = (unsigned short*)ImageRow( i );

				for( int j = 0; j < m_iCols; j++ )
				{
					//int iValue = ((psRow[j] - iMinPixel) * 256) / iDivisor;
					//unsigned int iValue = (psRow[j]>>6) & 0x3F; iValue = iValue<<2;
					unsigned int iValue = psRow[j] & 0xFF;//  iValue = iValue<<3;

					fputc( iValue, outfile );
				}
			}
		}
		else if( BitsPerPixel() == 8 )
		{
			fprintf( outfile, "P5\n%d %d\n%d\n", m_iCols, m_iRows, 255 );
			for( int i = 0; i < m_iRows; i++ )
			{
				fwrite( ImageRow( i ), 1, m_iCols, outfile );
			}
		}
		else
		{
			// Not working with anything but 24 & 8 bit images

			//fclose( outfile );
			return FALSE;
		}
	}
	else
	{
		if( BitsPerPixel() == 32 )
		{
			// Float
			fprintf( outfile, "P8\n%d %d\n%d\n", m_iCols, m_iRows, 0 );
			for( int i = 0; i < m_iRows; i++ )
			{
				fwrite( ImageRow( i ), 1, m_iCols*sizeof(float), outfile );
			}

		}
		if( BitsPerPixel() == 24 )
		{
			// For some screwy reason, RGB must be BGR for PPM format

			fprintf( outfile, "P3\n%d %d\n%d\n", m_iCols, m_iRows, 255 );

			for( int i = 0; i < m_iRows; i++ )
			{
				unsigned char *pucRow = ImageRow( i );
				for( int j = 0; j < m_iCols*3; j++ )
				{
					fprintf( outfile, "%d ", pucRow[j] );
				}
			}
		}
		else if( BitsPerPixel() == 16 )
		{
			// For now, lets not allow this...
			//assert( FALSE );
			//fclose( outfile );
			//return FALSE;

			//int iMaxPixel, iMinPixel;
			//int iDivisor = iMaxPixel - iMinPixel + 1;

			// Treat like a greyscale image, although this could be color
			fprintf( outfile, "P5\n%d %d\n%d\n", m_iCols, m_iRows, 255 );

			for( int i = 0; i < m_iRows; i++ )
			{
				unsigned short *psRow = (unsigned short*)ImageRow( i );

				for( int j = 0; j < m_iCols; j++ )
				{
					//int iValue = ((psRow[j] - iMinPixel) * 256) / iDivisor;
					//unsigned int iValue = (psRow[j]>>6) & 0x3F; iValue = iValue<<2;
					unsigned int iValue = psRow[j] & 0xFF;//  iValue = iValue<<3;

					fputc( iValue, outfile );
				}
			}
		}
		else if( BitsPerPixel() == 8 )
		{
			fprintf( outfile, "P5\n%d %d\n%d\n", m_iCols, m_iRows, 255 );
			for( int i = 0; i < m_iRows; i++ )
			{
				fwrite( ImageRow( i ), 1, m_iCols, outfile );
			}
		}
		else
		{
			// Not working with anything but 24 & 8 bit images

			//fclose( outfile );
			return FALSE;
		}
	}

	return TRUE;
}



int
GenericImage::WriteToFile(
	const char *pcFileName
	) const
{
	if( !pcFileName )
	{
		return FALSE;
	}

	FILE *outfile = fopen( pcFileName, "wb" );
	if( !outfile )
	{
		return FALSE;
	}

	int iReturn = WriteToFile( (void *)outfile );

	fclose( outfile );

	return iReturn;
}


int
GenericImage::MirrorImage(
						  )
{
	int iBytesPerPixel = m_iBitsPerPixel/8;
	for( int iRow = 0; iRow < Rows(); iRow++ )
	{
		unsigned char *pucRow = ImageRow(iRow);
		unsigned char ucMem[4];
		for( int iCol = 0; iCol < Cols()/2; iCol++ )
		{
			memcpy( ucMem, pucRow + iCol*iBytesPerPixel, iBytesPerPixel );
			memcpy( pucRow + iCol*iBytesPerPixel, pucRow + (Cols()-iCol-1)*iBytesPerPixel, iBytesPerPixel );
			memcpy( pucRow + (Cols()-iCol-1)*iBytesPerPixel, ucMem, iBytesPerPixel );
		}
	}
	return 1;
}


int
GenericImage::MirrorImageVertical(
	)
{
	int iBytesPerPixel = m_iBitsPerPixel/8;
	for( int iRow = 0; iRow < Rows()/2; iRow++ )
	{
		unsigned char *pucRowFirst = ImageRow(iRow);
		unsigned char *pucRowLast  = ImageRow(Rows()-1-iRow);
		unsigned char ucMemTemp[20];
		for( int iCol = 0; iCol < Cols(); iCol++ )
		{
			memcpy( &ucMemTemp, pucRowFirst+iCol*iBytesPerPixel, iBytesPerPixel );
			memcpy( pucRowFirst+iCol*iBytesPerPixel, pucRowLast+iCol*iBytesPerPixel, iBytesPerPixel );
			memcpy( pucRowLast+iCol*iBytesPerPixel, &ucMemTemp, iBytesPerPixel );
		}
	}
	return 1;
}

int
GenericImage::ExchangeRowsCols(
	)
{
	GenericImage ppImgNew;
	ppImgNew.Initialize( Cols(), Rows(), Rows()*BitsPerPixel()/8, BitsPerPixel() );
	int iBytesPerPixel = BitsPerPixel()/8;
	for( int iRow = 0; iRow < Rows(); iRow++ )
	{
		unsigned char ucMemTemp[20];
		for( int iCol = 0; iCol < Cols(); iCol++ )
		{
			unsigned char *pucRowOrig = ImageRow(iRow);
			unsigned char *pucRowNew = ppImgNew.ImageRow(iCol);

			for( int k = 0; k < iBytesPerPixel; k++ )
			{
				pucRowNew[iRow*iBytesPerPixel+k] = pucRowOrig[iCol*iBytesPerPixel+k];
			}
		}
	}

	Initialize( ppImgNew, 1 );

	return 1;
}

int
GenericImage::ReverseColors(
		)
{
	if( Rows() <= 0 || Cols() <= 0 )
	{
		return 0;
	}

	if( m_iBitsPerPixel != 24 )
	{
		// No need to reverse greyscale
		return 0;
	}

	typedef unsigned char RGBS[3];
	for( int r = 0; r < Rows(); r++ )
	{
		RGBS *prgbRow = (RGBS *)ImageRow(r);
		for( int c = 0; c < Cols(); c++ )
		{
			unsigned char chTemp0 = prgbRow[c][0];
			prgbRow[c][0] = prgbRow[c][2];
			prgbRow[c][2] = chTemp0;
		}
	}

	return 1;
}


#ifdef _DEBUG

int
GenericImage::Cols() const
{
	return m_iCols;
}

int
GenericImage::Rows() const
{
	return m_iRows;
}

int
GenericImage::RowInc() const
{
	return m_iRowInc;
}

int
GenericImage::BitsPerPixel() const
{
	return m_iBitsPerPixel;
}

unsigned char *
GenericImage::ImageRow( int iRow ) const
{
	assert( iRow >= 0 );

	return(m_pData + iRow*m_iRowInc);
}

int
GenericImage::SubSample(
		GenericImage &imgOut
	)
{
	int iBytesPerPixel = BitsPerPixel()/8;
	assert( imgOut.Rows() <= Rows()/2 );
	assert( imgOut.Cols() <= Cols()/2 );
	for( int r = 0; r < Rows()/2; r++ )
	{
		unsigned char *pucRow0 = ImageRow( 2*r+0 );
		unsigned char *pucRow1 = ImageRow( 2*r+1 );
		unsigned char *pucRowOut = ImageRow( r );
		for( int c = 0; c < Cols()/2; c++ )
		{
			for( int k = 0; k < iBytesPerPixel; k++ )
			{
				int iPix = pucRow0[iBytesPerPixel*c+k]
				+pucRow0[iBytesPerPixel*(c+1)+k]
				+pucRow1[iBytesPerPixel*c+k]
				+pucRow1[iBytesPerPixel*(c+1)+k];
				pucRowOut[c+k] = iPix/4;
			}
		}
	}
	return 1;
}

#endif
