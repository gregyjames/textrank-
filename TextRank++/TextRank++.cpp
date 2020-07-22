// TextRank++.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include "boost/multi_array.hpp"
#include "boost/algorithm/string.hpp"
#include "NumCpp.hpp"

using namespace std;
using namespace boost;
using namespace nc;

double cosine_similarity(double *a, double *b, unsigned int vector_Length)
{
	double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
	for (unsigned int i = 0u; i < vector_Length; ++i) {
		dot += a[i] * b[i];
		denom_a += a[i] * a[i];
		denom_b += b[i] * b[i];
	}
	return dot / (sqrt(denom_a) * sqrt(denom_b));
}

double similarity(string sentence1, string sentence2)
{
	cout << sentence1 << "->" << sentence2 << "->";
	auto sent1lower = to_lower_copy(sentence1);
	auto sent2lower = to_lower_copy(sentence2);

	vector<string> sent1_words;
	vector<string> sent2_words;

	split(sent1_words,sent1lower, is_any_of(" "));
	split(sent2_words,sent2lower, is_any_of(" "));

	vector<string> words_total;
	sort(sent1_words.begin(), sent1_words.end());
	sort(sent2_words.begin(), sent2_words.end());
	set_union(sent1_words.begin(), sent1_words.end(), sent2_words.begin(), sent2_words.end(), back_inserter(words_total));

	const int total_word_count = words_total.size();

	vector<double> wordOccur1;
	vector<double> wordOccur2;

	wordOccur1.resize(total_word_count,0);
	wordOccur2.resize(total_word_count, 0);

	for(auto i = 0; i < sent1_words.size(); i++)
	{
		auto it = std::find(words_total.begin(), words_total.end(), sent1_words[i]);
		const int index = it - words_total.begin();
		wordOccur1[index] += 1;
	}

	for (auto i = 0; i < sent2_words.size(); i++)
	{
		auto it = std::find(words_total.begin(), words_total.end(), sent2_words[i]);
		const int index = it - words_total.begin();
		wordOccur2[index] += 1;
	}

	const auto s = cosine_similarity(wordOccur1.data(), wordOccur2.data(), total_word_count);
	cout << s << endl;
	return s;
}

typedef multi_array<double, 2> array_type;
typedef multi_array<double, 1> ones;

NdArray<double> textRank(vector<string> sentences, double eps = 0.0001, double d = 0.85)
{
	//set up matrix
	const auto N = sentences.size();
	auto P = divide(nc::ones<double>(Shape(N, 1)), static_cast<double>(N));
	auto matrix = nc::zeros<double>(Shape(N, N));

	try
	{
		for (auto i = 0; i < N; i++)
		{
			for (auto j = 0; j < N; j++)
			{
				if (i == j)
				{
					matrix.at(i, j) = 0;
				}
				else
				{
					matrix.at(i, j) = similarity(sentences[i], sentences[j]);
				}
			}
		}

		bool done = false;

		//matrix.print();

		while (done == false)
		{
			//try
			//{
			auto new_p = nc::ones<double>(Shape(N, 1));
			auto multiplied = multiply(new_p, (1 - d) / N);
			auto dot = multiply(nc::dot(matrix, P), d);
			auto value = add(multiplied, dot);
			auto subtracted = subtract(value, P);
			auto absolute = toStlVector(abs(subtracted));
			const auto delta = accumulate(absolute.begin(), absolute.end(), 0.0);

			if (delta <= eps)
			{
				// ReSharper disable once CppAssignedValueIsNeverUsed
				done = true;
				break;
			}
			else if(delta == INFINITY)
			{
				break;
			}
			else
			{
				cout << "Delta -> " << delta << endl;
				copyto(P, value);
			}
		}
	}catch (const std::exception& exception)
	{
		cout << exception.what() << endl;
	}
	
	return P;

}

int main()
{
	string input = "The characteristic of poor information of short text often makes the effect of traditional keywords extraction not as good as expected. In this paper we propose a graphbased ranking algorithm by exploiting Wikipedia as an external knowledge base for short text keywords extraction. To overcome the shortcoming of poor information of short text we introduce the Wikipedia to enrich the short text. We regard each entry of Wikipedia as a concept therefore the semantic information of each word can be represented by the distribution of Wikipedia's concept. And we measure the similarity between words by constructing the concept vector. Finally we construct keywords matrix and use TextRank for keywords extraction. The comparative experiments with traditional TextRank and baseline algorithm show that our method gets better precision recall and F-measure value. It is shown that TextRank by exploiting Wikipedia is more suitable for short text keywords extraction.";
	auto s = vector<string>();
	split(s, input, is_any_of("."));
	
	cout.precision(2);
	auto result = textRank(s, 0.0001, .85);
	map<double, string> dict;
	
	for(int i = 0; s.size() > i; i++)
	{
		dict[result[i]] = s[i];
	}

	for (map<double, string>::iterator it = dict.begin(); it != dict.end(); it++)
		cout << "(" << (*it).first << ", "
		<< (*it).second << ")" << endl;

	return 0;
}

