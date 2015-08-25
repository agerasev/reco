#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <ios>

unsigned int readUInt(std::istream &is)
{
	unsigned int var;
	(((var = is.get()<<24) |= is.get()<<16) |= is.get()<<8) |= is.get();
	return var;
}

class Image
{
public:
	
	int sx;
	int sy;
	
	int digit;
	float *data;
	
	Image(int sx, int sy)
	{
		data = new float[sx*sy];
	}
	
	~Image()
	{
		delete data;
	}
};

class ImageSet
{
public:
	
	int size;
	int size_x, size_y;
	
	Image **images;
	
	ImageSet(int s, int sx, int sy)
	{
		size = s;
		size_x = sx;
		size_y = sy;
		images = new Image*[size];
	}
	
	~ImageSet()
	{
		delete images;
	}
};

ImageSet *createImageSet(std::string labels, std::string images)
{
	std::ifstream ls(labels), is(images);
	if(!ls.is_open() || !is.is_open())
	{
		std::cerr << "files not found" << std::endl;
		ls.close();
		is.close();
		return nullptr;
	}
	
	int magic,lnum,inum,rows,cols;
	
	magic = readUInt(ls);
	lnum = readUInt(ls);
	
	if(magic != 2049)
	{
		std::cerr << labels << " is not a label file" << std::endl;
		return nullptr;
	}
	
	magic = readUInt(is);
	inum = readUInt(is);
	rows = readUInt(is);
	cols = readUInt(is);
	
	if(magic != 2051)
	{
		std::cerr << labels << " is not a image file" << std::endl;
		return nullptr;
	}
	
	if(lnum != inum)
	{
		std::cerr << "image and label numbers doesn't match" << std::endl;
		return nullptr;
	}
	
	ImageSet *set = new ImageSet(inum,cols,rows);
	
	char buf;
	for(int i = 0; i < inum; ++i)
	{
		set->images[i] = new Image(cols,rows);
		
		ls.get(buf);
		set->images[i]->digit = buf;
		
		for(int iy = 0; iy < rows; ++iy)
		{
			for(int ix = 0; ix < cols; ++ix)
			{
				is.get(buf);
				set->images[i]->data[iy*cols + ix] = float(static_cast<unsigned char>(buf))/255.0;
			}
		}
	}
	
	ls.close();
	is.close();
	
	return set;
}

void destroyImageSet(ImageSet *set)
{
	for(int i = 0; i < set->size; ++i)
	{
		delete set->images[i];
	}
	delete set;
}
