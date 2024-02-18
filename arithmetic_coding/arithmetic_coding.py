from bitstring import BitArray, Array
import numpy as np

class ArithmeticCode:
    def __init__(self, resolution=32):
        self.resolution = resolution
        self.high = 2**resolution - 1
        self.low = 0
        self.zero_bit_mask = 2**(resolution - 1)
        self.first_bit_mask = 2**(resolution - 2)
        self.emission_buffer = 0
        self.verbose = False

    def encode_symbols(self, cdfs, symbols):
        code = BitArray(0)
        for i in range(len(symbols)):
            code.append(self.encode_symbol(cdfs[i], symbols[i]))
        return code

    def encode_intervals(self, intervals):
        code = BitArray(0)
        for interval in intervals: code.append(self.encode_interval(interval))
        return code

    def encode_symbol(self, cdf, symbol):
        interval = (cdf[symbol], cdf[symbol + 1])
        return self.encode_interval(interval)

    def encode_interval(self, interval):
        cdf_low, cdf_high = interval
        high, low, emission_buffer, resolution = self.high, self.low, self.emission_buffer, self.resolution
        code = BitArray(0)
        verbose = self.verbose
        print(f"\n(high, low) = {(low, (high+1))}, with buffer {emission_buffer}") if verbose else None
        print(f"\ninterval = {(cdf_low, cdf_high)}") if verbose else None

        interval_length = (high+1) - low

        # Adjust high and low based on the cdf intervals, convert to integer arithmetic
        high = low + int(interval_length * cdf_high)-1
        low = low + int(interval_length * cdf_low)
        while (high & self.zero_bit_mask) == 0 or (low & self.zero_bit_mask) != 0:
            bitsegment = BitArray(emission_buffer + 1)
            bitsegment[0] = 1
            if (high & self.zero_bit_mask) == 0:
                code.append(~bitsegment)
            else:
                code.append(bitsegment)
            high <<= 1  # Shift left and add 1 as the least significant bit
            high += 1
            low <<= 1
            high %= 2**self.resolution
            low %= 2**self.resolution
            emission_buffer = 0
        # Adjust for underflow ranges
        while (high & self.first_bit_mask == 0) and (low & self.first_bit_mask != 0):
            emission_buffer += 1
            high <<= 1
            high |= (self.zero_bit_mask + 1)
            high %= 2**self.resolution
            low <<= 1
            low %= self.zero_bit_mask
        print(f"code = {code}") if self.verbose else None
        self.high, self.low, self.emission_buffer = high, low, emission_buffer
        return code

    def decode_symbols(self, cdfs, code):
        symbols = []
        for cdf in cdfs:
            symbol, code = self.decode_symbol(cdf, code)
            symbols.append(symbol)
        return symbols

    def decode_symbol(self, cdf, code):
        high, low, emission_buffer, resolution = (
            self.high,
            self.low,
            self.emission_buffer,
            self.resolution,
        )
        code_number = BitArray(resolution)
        if len(code) == 0:
            code_number[0] = 1
        else:
            code_number[0] = code[0]
            code_number_len = min(1 + len(code[emission_buffer+1:]), resolution)
            code_number[1:code_number_len] = code[
                emission_buffer + 1 : emission_buffer + code_number_len
            ]
            if code_number_len < resolution:
                code_number[code_number_len] = 1
        numerical_code_number = int(code_number.bin, 2)
        probability = (numerical_code_number - low) / ((high+1) - low)
        verbose = self.verbose
        print(f"\ninterval = {(low, numerical_code_number, (high+1))}") if self.verbose else None
        print(f"probability = {probability}") if verbose else None

        # Iterate through the CDF to find the symbol
        # The CDF is an ascending numpy array,
        # `symbol` is defined by cdf[symbol] <= probability < cdf[symbol+1]
        symbol = np.searchsorted(cdf, probability, side='right') - 1
        assert cdf[symbol] <= probability < cdf[symbol + 1]
        interval = (cdf[symbol], cdf[symbol + 1])
        code_prefix = self.encode_interval(interval)
        print(f"Decoded symbol {symbol}") if verbose else None
        print(f"Decoded code prefix {code_prefix}") if verbose else None

        return symbol, code[len(code_prefix):]