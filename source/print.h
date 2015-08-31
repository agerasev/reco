#pragma once

#include <iostream>
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

void printLayer(const Layer *layer)
{
	std::cout << layer->getInput() << std::endl;
	std::cout << layer->getOutput() << std::endl;
}

void printConn(const Conn *conn)
{
	std::cout << conn->getWeight() << std::endl;
	std::cout << conn->getBias() << std::endl;
}

void printLayerError(const Layer_BP *layer)
{
	std::cout << layer->getInputError() << std::endl;
	std::cout << layer->getOutputError() << std::endl;
}

void printConnGrad(const Conn_BP *conn)
{
	std::cout << conn->getWeightGrad() << std::endl;
	std::cout << conn->getBiasGrad() << std::endl;
}
