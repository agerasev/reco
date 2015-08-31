#pragma once

#include <string>
#include <iostream>
#include <cstdio>

#include <vector>
#include <exception>

static unsigned int readUInt(FILE *f)
{
	unsigned int var;
	(((var = fgetc(f)<<24) |= fgetc(f)<<16) |= fgetc(f)<<8) |= fgetc(f);
	return var;
}

class Image
{
private:
	int sx;
	int sy;
	
	int digit;
	std::vector<float> data;
	
public:
	void setSize(int size_x, int size_y)
	{
		sx = size_x;
		sy = size_y;
		data.resize(sx*sy);
	}
	
	void setDigit(int d)
	{
		digit = d;
	}
	
	std::vector<float> &getData()
	{
		return data;
	}
	
	int getSizeX() const
	{
		return sx;
	}
	
	int getSizeY() const
	{
		return sy;
	}
	
	int getDigit() const
	{
		return digit;
	}
	
	const std::vector<float> &getData() const
	{
		return data;
	}
};

class ImageSet
{
public:
	class Exception : public std::exception
	{
	private:
		std::string message;
		
	public:
		Exception(const std::string &msg)
		  : message(msg)
		{
			
		}
		
		virtual const char *what() const noexcept override
		{
			return message.data();
		}
	};
	
private:
	int size;
	int img_sx, img_sy;
	
	std::vector<Image> images;
	
	class File
	{
	public:
		FILE *file;
		
		File(const std::string &name)
		{
			file = fopen(name.data(), "rb");
			if(!file)
			{
				throw Exception("file " + name + " not found");
			}
		}
	};
	
public:
	ImageSet(const std::string &labels, const std::string &images)
	{
		File ls(labels), is(images);
		
		int magic,lnum,inum,rows,cols;
		
		magic = readUInt(ls.file);
		lnum = readUInt(ls.file);
		
		if(magic != 2049)
		{
			throw Exception(labels + " is not a label file");
		}
		
		magic = readUInt(is.file);
		inum = readUInt(is.file);
		rows = readUInt(is.file);
		cols = readUInt(is.file);
		
		if(magic != 2051)
		{
			throw Exception(labels + " is not a image file");
		}
		
		if(lnum != inum)
		{
			throw Exception("image and label numbers don't match");
		}
		
		size = inum;
		img_sx = cols;
		img_sy = rows;
		
		this->images.resize(inum);
		
		char buf;
		for(int i = 0; i < inum; ++i)
		{
			this->images[i].setSize(cols, rows);
			
			buf = fgetc(ls.file);
			this->images[i].setDigit(buf);
			
			for(int iy = 0; iy < rows; ++iy)
			{
				for(int ix = 0; ix < cols; ++ix)
				{
					buf = fgetc(is.file);
					this->images[i].getData()[iy*cols + ix] = float(static_cast<unsigned char>(buf))/255.0;
				}
			}
		}
	}
	
	int getImageSizeX() const
	{
		return img_sx;
	}
	
	int getImageSizeY() const
	{
		return img_sy;
	}
	
	int getSize() const
	{
		return size;
	}
	
	const std::vector<Image> &getImages() const
	{
		return images;
	}
};
