# simple script for converting Buckwalter to Arabic
# wrapping the Encode::Buckwalter perl module
# Author: Yonatan Belinkov, belinkov@mit.edu
# Date: April 1st, 2013

use encoding "utf-8";

use Encode;
use Encode::Buckwalter;

#$ref_utf = "\x{0641}\x{0648}\x{0628}\x{0x631}";
#$ref_bwt = "fwbr";

#$bwt_out = encode( "Buckwalter", $ref_utf );  # eq ref_bwt
#$utf_out = decode( "Buckwalter", $ref_bwt );  # eq ref_utf


#print $bwt_out;
#print $utf_out;

$bwt_in = @ARGV[0];

$utf_res = decode( "Buckwalter", $bwt_in );
print $utf_res;




