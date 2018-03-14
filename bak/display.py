
import pdb

# add style to raw. color: 'r', 'g', 'b', 'p'; 'p' means no color
def add_style(raw, color = 'r'):
    if raw == '\n': # newline symbol
        if color in ['r', 'g', 'b']:
            return '<' + color + '>' + '\\' + 'n<br>' + '</' + color + '>'
        elif color == 'p':
            return '\n'
        else:
            raise Exception('The color ' + color + ' is not available! Exit.')
            sys.exit(-1)
    elif raw.isprintable():
        if color in ['r', 'g', 'b']:
            return '<' + color + '>' + raw + '</' + color + '>'
        elif color == 'p':
            return raw
        else:
            raise Exception('The color ' + color + ' is not available! Exit.')
            sys.exit(-1)
    else:
        pdb.set_trace()
        raise Exception(raw + ' is not printable! Exit.')
        sys.exit(-1)

# write content to a html file
def write_html(filename, content):
    # define css style
    style = """<style type='text/css'>
    html {
      font-family: Courier;
    }
    r {
      color: #ff0000;
    }
    g {
      color: #00ff00;
    }
    b {
      color: #0000ff;
    }
    </style>"""
    with open(filename, 'w') as f:
        f.write('<html>\n')
        f.write(style + '\n\n')
        f.write(content)
        f.write('\n\n</html>')
    f.close()