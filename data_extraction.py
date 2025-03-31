import zstandard as zstd

input_file = "lichess_db_standard_rated_2013-01.pgn.zst"
output_file = "data.pgn"

dctx = zstd.ZstdDecompressor()

with open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
    with dctx.stream_reader(f_in) as reader:
        while chunk := reader.read(16384):  # Read in chunks
            f_out.write(chunk)

print("Decompression complete! 🎉")
