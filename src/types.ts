export type STSRecord = {
  sentence1: string;
  sentence2: string;
  score: number;
};

export type Embedding = number[];

export type STSDataset = {
  rows: {
    row: STSRecord
  }[];
  num_rows_total: number,
  num_rows_per_page: number,
}
