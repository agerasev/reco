#pragma once

#include <string>
#include <iostream>
#include <cstdio>

static unsigned int readUInt(FILE *f)
{
	unsigned int var;
	(((var = fgetc(f)<<24) |= fgetc(f)<<16) |= fgetc(f)<<8) |= fgetc(f);
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

ImageSet *createImageSet(const std::string &labels, const std::string &images)
{
	FILE *ls, *is;
	ls = fopen(labels.data(), "rb");
	if(!ls)
	{
		std::cerr << "file " << labels << " not found" << std::endl;
		return nullptr;
	}
	
	is = fopen(images.data(), "rb");
	if(!is)
	{
		std::cerr << "file " << labels << " not found" << std::endl;
		fclose(ls);
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
		
		buf = fgetc(ls);
		set->images[i]->digit = buf;
		
		for(int iy = 0; iy < rows; ++iy)
		{
			for(int ix = 0; ix < cols; ++ix)
			{
				buf = fgetc(is);
				set->images[i]->data[iy*cols + ix] = float(static_cast<unsigned char>(buf))/255.0;
			}
		}
	}
	
	fclose(ls);
	fclose(is);
	
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
