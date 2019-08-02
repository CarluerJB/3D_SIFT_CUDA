
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "PpImage.h"

#define FALSE 0
#define TRUE 1

PpImage::PpImage()
{
	m_cCode = 0;
}

PpImage::~PpImage()
{
}

int
PpImage::InitFromStream( const char *pcFileName )
{
	if( !pcFileName )
	{
		return FALSE;
	}
	
	FILE *infile = fopen( pcFileName, "rb" );
	if( !infile )
	{
		return FALSE;
	}
	
	unsigned char buffer[500];
	if( !fgets( (char*)buffer, sizeof( buffer ), infile )  )
	{
		fclose( infile );
		return FALSE;
	}
	
	if( buffer[0] != 'P' )
	{
		fclose( infile );
		return FALSE;
	}
	
	// Save image type code
	
	m_cCode = buffer[1];
	
	switch( m_cCode )
	{
	case '2':
		// ascii pgm
		m_iBitsPerPixel = 8;
		break;
	case '5':
		// binary pgm
		m_iBitsPerPixel = 8;
		break;
	case '3':
		// ascii ppm
		m_iBitsPerPixel = 24;
		break;
	case '6':
		// binary ppm
		m_iBitsPerPixel = 24;
		break;
	case '8':
		// binary pgm, float 
		m_iBitsPerPixel = 32;
		break;
	default:
		
		// Unrecognized format
		
		fclose( infile );
		return FALSE;
	}
	
	// Read past comment (for now)
	
	while( fgets( (char*)buffer, sizeof( buffer ), infile ) && strchr( (char*)buffer, '#' ) )
	{
	}
	
	// Scan for rows/cols
	
	if( sscanf( (char*)buffer, " %d %d", &m_iCols, &m_iRows )
		!= 2 )
	{
		fclose( infile );
		return FALSE;
	}
	
	m_iRowInc = m_iCols*(m_iBitsPerPixel/8);
	
	// Read next line
	
	if( !fgets( (char*)buffer, sizeof( buffer ), infile ) )
	{
		fclose( infile );
		return FALSE;
	}
	
	// Scan for max value
	
	if( sscanf( (char*)buffer, " %d", &m_iMaxPixelValue ) != 1 )
	{
		fclose( infile );
		return FALSE;
	}
	
	if( m_iMaxPixelValue > 255 )
	{
		fclose( infile );
		return FALSE;
	}
	
	// Reallocate image

	if( !Reallocate( m_iRows*m_iRowInc ) )
	{
		fclose( infile );
		return FALSE;
	}
	
	// Read data
	
	if( m_cCode == '5' || m_cCode == '6' )
	{
		// Binary data

		size_t nBytesRead;

		nBytesRead = fread( m_pData, 1, m_iRows*m_iRowInc, infile );
		
		if( nBytesRead != (size_t)m_iRows*m_iRowInc )
		{
			fclose( infile );
			return FALSE;
		}
	}
	else if( m_cCode == '2' || m_cCode == '3' )
	{
		// Ascii data - slow read
		
		int nBytesToRead = m_iRows*m_iRowInc;
		
		for( int i = 0; i < nBytesToRead; i++ )
		{
			int iPixel;

			if( fscanf( infile, "%d", &iPixel ) != 1 )
			{
				fclose( infile );
				return FALSE;
			}

			ImageRow( 0 )[i] = (unsigned char)iPixel;
		}
	}
	else if( m_cCode == '8' )
	{
		// Float data
		for( int i = 0; i < m_iRows; i++ )
		{
			fread( ImageRow( i ), 1, m_iCols*sizeof(float), infile );
		}
	}
	else
	{
		assert( 0 );
		fclose( infile );
		return FALSE;
	}

	// Switch bytes for 24-bit RGB image

	if( m_cCode == '3' || m_cCode == '6' )
	{
		// Must switch bytes
		//assert( FALSE );
	}

	fclose( infile );

	return TRUE;
}
