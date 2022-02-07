
int SetFlag(int flags, int flag, bool enabled)
{
	if (HasFlag(flags, flag))
	{
		if (!enabled)
			return flags -= flag;
	}
	else
	{
		if (enabled)
			return flags += flag;
	}

	return flags;
}

Body SetFlagB(Body body, int flag, bool enabled)
{
	Body out = body;

	if (HasFlag(out.Flag, flag))
	{
		if (!enabled)
			out.Flag -= flag;
	}
	else
	{
		if (enabled)
			out.Flag += flag;
	}

	return out;
}

bool HasFlag(int flags, int check)
{
	return (check & flags) != 0;
}

bool HasFlagB(Body body, int check)
{
	return (check & body.Flag) != 0;
}

// Compute number of blocks needed for the given length and threads per block.
int BlockCount(int len, int threads)
{
	int blocks = len / threads;
	int mod = len % threads;

	if (mod > 0)
		blocks += 1;

	return blocks;
}

// Is the specified cell a neighbor of the test cell?
bool IsFar(int2 cell, int2 testCell)
{
	uint2 diff = abs_diff(cell, testCell);
	if (diff.x > 1 || diff.y > 1)
		return true;

	return false;
}
