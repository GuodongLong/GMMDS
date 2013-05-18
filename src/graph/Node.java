package graph;

import java.util.ArrayList;
import java.util.List;

public class Node {

	public int docId = -1;
	public List<Integer> occurrence = new ArrayList<Integer>();
	public List<Node> neighbors = new ArrayList<Node>();
	
	public Node(int docid)
	{
		this.docId = docid;
	}
}
