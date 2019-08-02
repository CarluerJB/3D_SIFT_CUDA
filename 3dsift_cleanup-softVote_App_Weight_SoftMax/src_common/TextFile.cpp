
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "TextFile.h"

TextFile::TextFile(
				   )
{
	m_iLines = 0;
	m_iLineSize = 0;
	m_pcLines = 0;
}

TextFile::~TextFile(
					)
{
	Delete();
}

int
TextFile::Initialized(
	)
{
	if( m_iLines > 0 &&
		m_iLineSize > 0 &&
		m_pcLines != 0
		)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

int
TextFile::Delete(
			   )
{
	if( m_pcLines )
	{
		assert( Initialized() );
		if( !Initialized() )
		{
			return -1;
		}
		delete [] m_pcLines;
		m_pcLines = 0;
	}
	else
	{
		assert( !Initialized() );
		if( Initialized() )
		{
			return -1;
		}
	}

	return 0;
}

int
TextFile::Init(
			   int iLines,
			   int iLineSize
			   )
{
	Delete();

	assert( !Initialized() );
	if( Initialized() )
	{
		return -1;
	}

	assert( iLines > 0 );
	assert( iLineSize > 0 );
	if( iLines <= 0 || iLineSize <= 0 )
	{
		return -2;
	}

	m_pcLines = new char[iLines*iLineSize];
	m_iLines = iLines;
	m_iLineSize = iLineSize;

	memset( m_pcLines, 0, sizeof(char)*iLines*iLineSize );

	return 0;
}

int
TextFile::Read(
		char *pcFileName
			   )
{
	FILE *infile = fopen( pcFileName, "rt" );
	if( !infile )
	{
		return -2;
	}

	char *buffer = new char[2000];
	if( !buffer )
	{
		fclose( infile );
		return -3;
	}

	int iLineSize = 0;
	int iLines    = 0;
	while( fgets( buffer, 2000, infile ) != 0 )
	{
		int iLength = strlen( buffer ) + 1;
		if( iLength > iLineSize )
		{
			// Save max line size
			iLineSize = iLength;
		}
		iLines++;
	}

	if( Init( iLines, iLineSize ) != 0 )
	{
		// Could not initialize
		delete [] buffer;
		fclose( infile );
		return -1;
	}

	// Rewind, read in lines
	fseek( infile, 0, SEEK_SET );
	int iLine = 0;
	while( fgets( m_pcLines + iLine*m_iLineSize, m_iLineSize, infile ) != 0 )
	//while( fscanf( infile, "%s", m_pcLines + iLine*m_iLineSize ) == 1 )
	{
		// Remove newline character
		int iLength = strlen( m_pcLines + iLine*m_iLineSize );
		assert( iLength > 0 );
		m_pcLines[iLine*m_iLineSize + iLength - 1] = 0;
		iLine++;
	}

	assert( iLine == m_iLines );

	delete [] buffer;
	fclose( infile );

	return 0;
}

int
TextFile::Write(
		char *pcFileName
				)
{
	assert( Initialized() );
	if( !Initialized() )
	{
		return -1;
	}

	assert( pcFileName );
	if( pcFileName == 0 )
	{
		return -4;
	}

	FILE *outfile = fopen( pcFileName, "wt" );
	if( !outfile )
	{
		return -2;
	}

	for( int iLine = 0; iLine < m_iLines; iLine++ )
	{
		fputs( m_pcLines + iLine*m_iLineSize, outfile );
		fputs( "\n", outfile );
	}

	fclose( outfile );

	return 0;
}

char *
TextFile::GetLine(
				  int iLine
				  )
{
	assert( iLine >= 0 && iLine < m_iLines );
	assert( Initialized() );
	if(
		!( iLine >= 0 && iLine < m_iLines ) ||
		!Initialized()
		)
	{
		return 0;
	}

	return m_pcLines + iLine*m_iLineSize;
}

int
TextFile::Lines(
		) const
{
	return m_iLines;
}

int
TextFile::LineSize(
		)
{
	return m_iLineSize;
}

char *
TextFile::GetFileName(
		char *pcPathFile
	)
{
	assert( pcPathFile );
	if( !pcPathFile )
	{
		return 0;
	}

	char *pchCurr = pcPathFile;
	char *pchNext = pcPathFile;

	if( pchCurr = strrchr( pcPathFile, '\\' ) )
	{
		return pchCurr+1;
	}
	else if( pchCurr = strrchr( pcPathFile, '/' ) )
	{
		return pchCurr+1;
	}
	else
	{
		return pcPathFile;
	}

	// Old code - did not work with Linux
	//while( pchNext = strchr( pchCurr, '\\' ) )
	//{
	//	pchCurr = pchNext + 1;
	//}
	//return pchCurr;
}

int
TextFile::SetDirectory(
		char *pcDirName
	)
{
	if( !pcDirName )
	{
		return -1;
	}
	int iDirNameLen = strlen( pcDirName );
	if( iDirNameLen < 0 )
	{
		return -2;
	}
	if( pcDirName[iDirNameLen-1] != '\\' )
	{
		iDirNameLen++;
	}

	// Calculate maximum file name length, without directory
	unsigned int uiMaxFileNameLen = 0;
	for( int i = 0; i < m_iLines; i++ )
	{
		char *pcFileName = GetFileName( GetLine(i) );
		assert( pcFileName );
		if( strlen( pcFileName ) > uiMaxFileNameLen )
		{
			uiMaxFileNameLen = strlen( pcFileName );
		}
	}

	int iLinesNew = m_iLines;
	int iLineSizeNew = iDirNameLen + uiMaxFileNameLen + 1;
	char *pcLinesNew = new char[iLinesNew*iLineSizeNew];
	if( !pcLinesNew )
	{
		return -3;
	}
	memset( pcLinesNew, 0, iLinesNew*iLineSizeNew );

	// Create new array with new directory names
	for( int i = 0; i < iLinesNew; i++ )
	{
		char *pcFileDirNew = pcLinesNew + i*iLineSizeNew;
		char *pcFileName = GetFileName( GetLine(i) );
		assert( pcFileName );

		if( pcDirName[iDirNameLen-1] == '\\' )
		{
			sprintf( pcFileDirNew, "%s%s", pcDirName, pcFileName );
		}
		else
		{
			sprintf( pcFileDirNew, "%s\\%s", pcDirName, pcFileName );
		}
	}

	Delete();

	m_iLines = iLinesNew;
	m_iLineSize = iLineSizeNew;
	m_pcLines = pcLinesNew;

	return 0;
}


int
TextFile::InitLinePointers(
		int &iNonZeroLines,
		char **ppcNonZeroLines
	)
{
	iNonZeroLines = 0;

	for( int i = 0; i < m_iLines; i++ )
	{
		char *pcFileName = GetLine(i);
		assert( pcFileName );
		if( strlen( pcFileName ) > 0 )
		{
			ppcNonZeroLines[iNonZeroLines] = pcFileName;
			iNonZeroLines++;
		}
	}

	return 0;
}
