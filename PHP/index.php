<?php
/* XML Header */
header("Content-Type:text/xml");

/* Read out variables */
$dataset = 0;
$message = "";
if(isset($_GET['dataset'])) $dataset = $_GET['dataset'];
if(isset($_GET['message'])) $message = $_GET['message'];
if ($dataset) {
	$dataset = (int) $dataset;
}
/* Run python script */
exec("python sentiment.py $dataset \"$message\"", $output);

/* Create XML File */
$results = array();
if (count($output) != 6 ) {
	/* BAD REQUEST */
	$results [] = array(
		'status' => 'BAD REQUEST',
		'dataset' => 'unknown',
		'message' => 'unknown',
		'sentiment' => 'unknown',
		'accuracy' => 'unknown',
		'precision' => 'unknown',
		'recall' => 'unknown'
	);
} else {
	/* GOOD REQUEST */
	$results [] = array(
		'status' => 'GOOD REQUEST',
		'dataset' => $output[0],
		'message' => $output[1],
		'sentiment' => $output[2],
		'accuracy' => $output[3],
		'precision' => $output[4],
		'recall' => $output[5]
	);
}
$doc = new DOMDocument("1.0");
$doc->formatOutput = true;

$root = $doc->createElement("results");
$root = $doc->appendChild($root);

foreach($results as $result) {
	$res = $doc->createElement("result");

	foreach( array_keys($result) as $key) {
		$element = $doc->createElement($key);
		$element->appendChild( $doc->createTextNode( $result[$key] ) );
		$res->appendChild($element);	   
	}
	$root->appendChild($res);
}
echo $doc->saveXML();
?>
