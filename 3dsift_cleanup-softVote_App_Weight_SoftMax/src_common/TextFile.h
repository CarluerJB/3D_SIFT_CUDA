
#ifndef __TEXTFILE_H__
#define __TEXTFILE_H__

class TextFile
{
public:
	
	TextFile(
		);
	
	~TextFile(
		);

	int
	Init(
		int iLines,
		int iLineSize
		);
	
	virtual int
	Read(
		char *pcFileName
		);
	
	virtual int
	Write(
		char *pcFileName
		);
	
	char *
	GetLine(
		int iLine
		);
	
	// Returns the number of lines
	int
	Lines(
		) const;

	// Returns the line size, equal to the length of the longest text line
	int
	LineSize(
		);

	//
	// Validate()
	//
	// After initalized, this function validates the data, can be
	// overridden for special purpose text files.
	//
	virtual int
	Validate(
	) { return 1; };

	//
	// SetDirectory()
	//
	// Changes the directory before all file names.
	//
	int
	SetDirectory(
		char *pcDirName
	);

	//
	// InitLinePointers()
	//
	// Initialize an array of pointers, to each text line. This is used
	// to convert the string into command line format (int argc, char **argv).
	// Initializes only lines that contain non-whitespace text.
	// ppcNonZeroLines should contain enough space for all lines, i.e.
	// char **ppcNonZeroLines = new char**[ this->Lines() ].
	//
	//
	int
	InitLinePointers(
		int &iNonZeroLines,
		char **ppcNonZeroLines
	);

	char *
	GetFileName(
		char *pcPathFile
		);

protected:
	
	int
	Delete(
		);

	int
	Initialized(
	);
	
	int		m_iLines;
	int		m_iLineSize;
	char	*m_pcLines;
};

#endif
