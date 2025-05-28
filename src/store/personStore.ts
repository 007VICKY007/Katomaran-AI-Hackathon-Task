import create from 'zustand';

export interface Person {
  id: string;
  name: string;
  image?: string;
  timestamp: string;
  created_at: Date;
}

interface PersonStore {
  people: Person[];
  setPeople: (people: Person[]) => void;
}

export const usePersonStore = create<PersonStore>((set) => ({
  people: [],
  setPeople: (people) => set({ people }),
}));