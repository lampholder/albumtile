Uses Apple Music and album.links to generate physical printable album tiles.

To use:

- download Inter.zip from Google fonts
- put it adjacent to albumtile.py
- do all the python do (virtualenv, `pip install -r requirements.txt`)
- `python albumtile.py https://music.apple.com/gb/album/ok-computer/1097861387`

That'll make you an a5 printable tile to wrap a 3 x 5 inch (index card-sized) lump of 3mm greyboard.

You can then use `concat_tile.py` to concatenate a pile of a5 pdfs into a single pdf full of a4 sheets of tiles.
