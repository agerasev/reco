#pragma once

#include <ostream>

#include <nn/buffer.hpp>
#include <nn/bp/layer.hpp>
#include <nn/bp/conn.hpp>

std::ostream &operator << (std::ostream &os, const Buffer &buffer)
{
	int size = buffer.getSize();
	float *data = new float[size];
	buffer.read(data);
	// os << buffer.isValid() << ":  ";
	for(int i = 0; i < size; ++i)
		os << data[i] << ' ';
	delete[] data;
	return os;
}

std::ostream &operator << (std::ostream &os, const Layer &layer)
{
	os << layer.getInput() << std::endl;
	os << layer.getOutput() << std::endl;
	return os;
}

std::ostream &operator << (std::ostream &os, const Conn &conn)
{
	os << conn.getWeight() << std::endl;
	os << conn.getBias() << std::endl;
	return os;
}

std::ostream &operator << (std::ostream &os, const Layer_BP &layer)
{
	os << static_cast<const Layer &>(layer);
	os << layer.getInputError() << std::endl;
	os << layer.getOutputError() << std::endl;
	return os;
}

std::ostream &operator << (std::ostream &os, const Conn_BP &conn)
{
	os << static_cast<const Conn &>(conn);
	os << conn.getWeightGrad() << std::endl;
	os << conn.getBiasGrad() << std::endl;
	return os;
}
